import argparse
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
from models import GAT, compute_acc
import bifeat
import os

torch.manual_seed(25)


def evaluate(model, g, labels, val_nid, test_nid, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(g, batch_size)
    model.train()
    return compute_acc(pred[val_nid],
                       labels[val_nid]), compute_acc(pred[test_nid],
                                                     labels[test_nid])


def run(rank, world_size, data, args):
    device = torch.cuda.current_device()
    # Unpack data
    train_nid, val_nid, test_nid, n_classes, g = data
    fan_out = [int(fanout) for fanout in args.fan_out.split(",")]
    sampler = dgl.dataloading.NeighborSampler(
        fan_out,
        prefetch_node_feats=["features"],
        prefetch_labels=["labels"],
    )
    dataloader = dgl.dataloading.DataLoader(g,
                                            train_nid,
                                            sampler,
                                            device=device,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            drop_last=False,
                                            num_workers=0,
                                            use_ddp=True,
                                            use_uva=True)
    # Define model and optimizer
    gat_heads = [int(head) for head in args.heads.split(",")]
    model = GAT(g.ndata["features"].shape[1],
                args.num_hidden,
                n_classes,
                len(fan_out),
                gat_heads,
                activation=F.relu,
                feat_dropout=args.feat_dropout,
                attn_dropout=args.attn_dropout)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[rank],
                                                output_device=rank)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    epoch_time_log = []
    sample_time_log = []
    load_time_log = []
    forward_time_log = []
    backward_time_log = []
    update_time_log = []
    for epoch in range(args.num_epochs):

        sample_time = 0
        load_time = 0
        forward_time = 0
        backward_time = 0
        update_time = 0
        num_seeds = 0
        num_inputs = 0
        num_iters = 0

        if args.breakdown:
            dist.barrier()
            torch.cuda.synchronize()
        epoch_tic = time.time()
        sample_begin = time.time()
        model.train()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            num_iters += 1
            if args.breakdown:
                dist.barrier()
                torch.cuda.synchronize()
            sample_time += time.time() - sample_begin

            load_begin = time.time()
            batch_inputs = blocks[0].srcdata["features"]
            batch_labels = blocks[-1].dstdata["labels"]
            batch_labels = batch_labels.long()
            num_seeds += len(blocks[-1].dstdata[dgl.NID])
            num_inputs += len(blocks[0].srcdata[dgl.NID])
            if args.breakdown:
                dist.barrier()
                torch.cuda.synchronize()
            load_time += time.time() - load_begin

            forward_start = time.time()
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            if args.breakdown:
                dist.barrier()
                torch.cuda.synchronize()
            forward_time += time.time() - forward_start

            backward_begin = time.time()
            optimizer.zero_grad()
            loss.backward()
            if args.breakdown:
                dist.barrier()
                torch.cuda.synchronize()
            backward_time += time.time() - backward_begin

            update_start = time.time()
            optimizer.step()
            if args.breakdown:
                dist.barrier()
                torch.cuda.synchronize()
            update_time += time.time() - update_start

            if (step + 1) % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = (torch.cuda.max_memory_allocated() /
                                 1000000 if torch.cuda.is_available() else 0)
                print("Part {} | Epoch {:05d} | Step {:05d} | Loss {:.4f} | "
                      "Train Acc {:.4f} | GPU {:.1f} MB".format(
                          rank, epoch, step + 1, loss.item(), acc.item(),
                          gpu_mem_alloc))
                train_acc_tensor = torch.tensor([acc.item()]).cuda()
                dist.all_reduce(train_acc_tensor, dist.ReduceOp.SUM)
                train_acc_tensor /= world_size
                if rank == 0:
                    print("Avg train acc {:.4f}".format(
                        train_acc_tensor[0].item()))

            if args.breakdown:
                dist.barrier()
                torch.cuda.synchronize()
            sample_begin = time.time()

        epoch_toc = time.time()

        for i in range(args.num_trainers):
            dist.barrier()
            if i == rank % args.num_trainers:
                timetable = ("=====================\n"
                             "Part {}, Epoch Time(s): {:.4f}\n"
                             "Sampling Time(s): {:.4f}\n"
                             "Loading Time(s): {:.4f}\n"
                             "Forward Time(s): {:.4f}\n"
                             "Backward Time(s): {:.4f}\n"
                             "Update Time(s): {:.4f}\n"
                             "#seeds: {}\n"
                             "#inputs: {}\n"
                             "#iterations: {}\n"
                             "=====================".format(
                                 rank,
                                 epoch_toc - epoch_tic,
                                 sample_time,
                                 load_time,
                                 forward_time,
                                 backward_time,
                                 update_time,
                                 num_seeds,
                                 num_inputs,
                                 num_iters,
                             ))
                print(timetable)
        sample_time_log.append(sample_time)
        load_time_log.append(load_time)
        forward_time_log.append(forward_time)
        backward_time_log.append(backward_time)
        update_time_log.append(update_time)
        epoch_time_log.append(epoch_toc - epoch_tic)

        if (epoch + 1) % args.eval_every == 0:
            tic = time.time()
            val_acc, test_acc = evaluate(
                model.module,
                g,
                g.ndata["labels"],
                val_nid,
                test_nid,
                args.batch_size_eval,
            )
            print("Part {}, Val Acc {:.4f}, Test Acc {:.4f}, time: {:.4f}".
                  format(rank, val_acc, test_acc,
                         time.time() - tic))
            acc_tensor = torch.tensor([val_acc, test_acc]).cuda()
            dist.all_reduce(acc_tensor, dist.ReduceOp.SUM)
            acc_tensor /= world_size
            if rank == 0:
                print("All parts avg val acc {:.4f}, test acc {:.4f}".format(
                    acc_tensor[0].item(), acc_tensor[1].item()))

    avg_epoch_time = np.mean(epoch_time_log[2:])
    avg_sample_time = np.mean(sample_time_log[2:])
    avg_load_time = np.mean(load_time_log[2:])
    avg_forward_time = np.mean(forward_time_log[2:])
    avg_backward_time = np.mean(backward_time_log[2:])
    avg_update_time = np.mean(update_time_log[2:])

    for i in range(args.num_trainers):
        dist.barrier()
        if i == rank % args.num_trainers:
            timetable = ("=====================\n"
                         "Part {}, Avg Time:\n"
                         "Epoch Time(s): {:.4f}\n"
                         "Sampling Time(s): {:.4f}\n"
                         "Loading Time(s): {:.4f}\n"
                         "Forward Time(s): {:.4f}\n"
                         "Backward Time(s): {:.4f}\n"
                         "Update Time(s): {:.4f}\n"
                         "=====================".format(
                             rank,
                             avg_epoch_time,
                             avg_sample_time,
                             avg_load_time,
                             avg_forward_time,
                             avg_backward_time,
                             avg_update_time,
                         ))
            print(timetable)
    all_reduce_tensor = torch.tensor([0], device="cuda", dtype=torch.float32)

    all_reduce_tensor[0] = avg_epoch_time
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    all_reduce_epoch_time = all_reduce_tensor[0].item() / world_size

    all_reduce_tensor[0] = avg_sample_time
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    all_reduce_sample_time = all_reduce_tensor[0].item() / world_size

    all_reduce_tensor[0] = avg_load_time
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    all_reduce_load_time = all_reduce_tensor[0].item() / world_size

    all_reduce_tensor[0] = avg_forward_time
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    all_reduce_forward_time = all_reduce_tensor[0].item() / world_size

    all_reduce_tensor[0] = avg_backward_time
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    all_reduce_backward_time = all_reduce_tensor[0].item() / world_size

    all_reduce_tensor[0] = avg_update_time
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    all_reduce_update_time = all_reduce_tensor[0].item() / world_size

    if rank == 0:
        timetable = ("=====================\n"
                     "All reduce time:\n"
                     "Throughput(seeds/sec): {:.4f}\n"
                     "Epoch Time(s): {:.4f}\n"
                     "Sampling Time(s): {:.4f}\n"
                     "Loading Time(s): {:.4f}\n"
                     "Forward Time(s): {:.4f}\n"
                     "Backward Time(s): {:.4f}\n"
                     "Update Time(s): {:.4f}\n"
                     "=====================".format(
                         train_nid.shape[0] / all_reduce_epoch_time,
                         all_reduce_epoch_time,
                         all_reduce_sample_time,
                         all_reduce_load_time,
                         all_reduce_forward_time,
                         all_reduce_backward_time,
                         all_reduce_update_time,
                     ))
        print(timetable)


def main(args):
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == args.num_trainers
    omp_thread_num = os.cpu_count() // args.num_trainers
    torch.cuda.set_device(rank)
    torch.set_num_threads(omp_thread_num)
    print("Set device to {} and cpu threads num {}".format(
        rank, omp_thread_num))
    shm_manager = bifeat.shm.ShmManager(rank,
                                        args.num_trainers,
                                        args.root,
                                        args.dataset,
                                        pin_memory=False)

    if args.dataset == "friendster":
        with_feature = False
        with_valid = False
        with_test = False
        feat_dtype = torch.float32
    elif args.dataset == "mag240M":
        with_feature = False
        with_valid = True
        with_test = True
        feat_dtype = torch.float16
    elif args.dataset == "ogbn-papers100M":
        with_feature = False
        with_valid = True
        with_test = True
        feat_dtype = torch.float32
    else:
        with_feature = True
        with_valid = True
        with_test = True
    g, metadata = shm_manager.load_dataset(with_feature=with_feature,
                                           with_test=with_test,
                                           with_valid=with_valid)
    dgl_g = dgl.graph(('csc', (g["indptr"], g["indices"], torch.tensor([]))))
    dgl_g.ndata["labels"] = g["labels"]
    if with_feature:
        dgl_g.ndata["features"] = g["features"]
    else:
        if shm_manager._is_chief:
            fake_feat = torch.randn(
                (metadata["num_nodes"], ),
                dtype=feat_dtype).reshape(-1,
                                          1).repeat(1, metadata["feature_dim"])
            g["features"] = shm_manager.create_shm_tensor(
                args.dataset + "_shm_features", feat_dtype, fake_feat.shape)
            g["features"].copy_(fake_feat)
            del fake_feat
        else:
            g["features"] = shm_manager.create_shm_tensor(
                args.dataset + "_shm_features", None, None)
    dist.barrier()
    dgl_g.ndata["features"] = g["features"]
    train_nid = g["train_idx"]
    if with_valid:
        val_nid = g["valid_idx"]
    else:
        val_nid = torch.tensor([]).long()
    if with_test:
        test_nid = g["test_idx"]
    else:
        test_nid = torch.tensor([]).long()
    print("start")
    data = train_nid, val_nid, test_nid, metadata["num_classes"], dgl_g

    run(rank, world_size, data, args)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        "Train nodeclassification GraphSAGE model")
    argparser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-products",
        help="datasets: ogbn-products, ogbn-papers100M, friendster, mag240M",
    )
    argparser.add_argument("--root", type=str, default="/data")
    argparser.add_argument(
        "--num-trainers",
        type=int,
        default="8",
        help=
        "number of trainers participated in the compress, no greater than available GPUs num"
    )
    argparser.add_argument("--lr", type=float, default=0.003)
    argparser.add_argument("--batch-size", type=int, default=1000)
    argparser.add_argument("--batch-size-eval", type=int, default=100000)
    argparser.add_argument("--log-every", type=int, default=20)
    argparser.add_argument("--eval-every", type=int, default=5)
    argparser.add_argument("--fan-out", type=str, default="5,10,15")
    argparser.add_argument("--num-hidden", type=int, default=32)
    argparser.add_argument("--heads", type=str, default="8,8,1")
    argparser.add_argument("--feat_dropout", type=float, default=0.2)
    argparser.add_argument("--attn_dropout", type=float, default=0.2)
    argparser.add_argument("--num-epochs", type=int, default=20)
    argparser.add_argument("--breakdown", action="store_true", default=False)
    args = argparser.parse_args()
    print(args)
    main(args)
