import argparse
import time

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
from models import GAT, compute_acc
import torch.distributed as dist
import torch

import sys

sys.path.append("utils")
from load_graph import load_ogb, load_reddit

torch.manual_seed(25)


def evaluate(model, g, labels, val_nid, test_nid, batch_size):
    model.eval()
    with th.no_grad():
        pred = model.inference(g, batch_size)
    model.train()
    return compute_acc(pred[val_nid],
                       labels[val_nid]), compute_acc(pred[test_nid],
                                                     labels[test_nid])


def run(rank, world_size, data, args):
    torch.cuda.set_device(rank)
    device = torch.device("cuda")
    dist.init_process_group('nccl',
                            'tcp://127.0.0.1:12347',
                            world_size=world_size,
                            rank=rank)
    # Unpack data
    train_nid, val_nid, test_nid, n_classes, g = data
    shuffle = True
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
                                            shuffle=shuffle,
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

    iter_tput = []
    epoch_time_log = []
    sample_time_log = []
    load_time_log = []
    forward_time_log = []
    backward_time_log = []
    update_time_log = []
    epoch = 0
    for epoch in range(args.num_epochs):
        tic = time.time()

        sample_time = 0
        load_time = 0
        forward_time = 0
        backward_time = 0
        update_time = 0
        num_seeds = 0
        num_inputs = 0

        with model.join():
            # Loop over the dataloader to sample the computation dependency
            # graph as a list of blocks.
            step_time = []
            if args.breakdown:
                dist.barrier()
                torch.cuda.synchronize()
            tic = time.time()
            tic_step = time.time()
            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                if args.breakdown:
                    dist.barrier()
                    torch.cuda.synchronize()
                sample_time += time.time() - tic_step

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

                step_t = time.time() - tic_step
                step_time.append(step_t)
                iter_tput.append(len(blocks[-1].dstdata[dgl.NID]) / step_t)
                tic_step = time.time()

                if (step + 1) % args.log_every == 0:
                    acc = compute_acc(batch_pred, batch_labels)
                    gpu_mem_alloc = (torch.cuda.max_memory_allocated() /
                                     1000000
                                     if torch.cuda.is_available() else 0)
                    print(
                        "Part {} | Epoch {:05d} | Step {:05d} | Loss {:.4f} | "
                        "Train Acc {:.4f} | GPU {:.1f} MB".format(
                            rank, epoch, step + 1, loss.item(), acc.item(),
                            gpu_mem_alloc))
                    train_acc_tensor = torch.tensor([acc.item()]).cuda()
                    dist.all_reduce(train_acc_tensor, dist.ReduceOp.SUM)
                    train_acc_tensor /= world_size
                    if rank == 0:
                        print("Avg train acc {:.4f}".format(
                            train_acc_tensor[0].item()))

        toc = time.time()
        epoch += 1

        for i in range(args.num_gpus):
            dist.barrier()
            if i == dist.get_rank() % args.num_gpus:
                timetable = ("=====================\n"
                             "Part {}, Epoch Time(s): {:.4f}\n"
                             "Sampling Time(s): {:.4f}\n"
                             "Loading Time(s): {:.4f}\n"
                             "Forward Time(s): {:.4f}\n"
                             "Backward Time(s): {:.4f}\n"
                             "Update Time(s): {:.4f}\n"
                             "#seeds: {}\n"
                             "#inputs: {}\n"
                             "=====================".format(
                                 dist.get_rank(),
                                 toc - tic,
                                 sample_time,
                                 load_time,
                                 forward_time,
                                 backward_time,
                                 update_time,
                                 num_seeds,
                                 num_inputs,
                             ))
                print(timetable)
        sample_time_log.append(sample_time)
        load_time_log.append(load_time)
        forward_time_log.append(forward_time)
        backward_time_log.append(backward_time)
        update_time_log.append(update_time)
        epoch_time_log.append(toc - tic)

        if epoch % args.eval_every == 0:
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

    for i in range(args.num_gpus):
        th.distributed.barrier()
        if i == th.distributed.get_rank() % args.num_gpus:
            timetable = ("=====================\n"
                         "Part {}, Avg Time:\n"
                         "Epoch Time(s): {:.4f}\n"
                         "Sampling Time(s): {:.4f}\n"
                         "Loading Time(s): {:.4f}\n"
                         "Forward Time(s): {:.4f}\n"
                         "Backward Time(s): {:.4f}\n"
                         "Update Time(s): {:.4f}\n"
                         "=====================".format(
                             th.distributed.get_rank(),
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
    all_reduce_epoch_time = all_reduce_tensor[0].item() / dist.get_world_size()

    all_reduce_tensor[0] = avg_sample_time
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    all_reduce_sample_time = all_reduce_tensor[0].item() / dist.get_world_size(
    )

    all_reduce_tensor[0] = avg_load_time
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    all_reduce_load_time = all_reduce_tensor[0].item() / dist.get_world_size()

    all_reduce_tensor[0] = avg_forward_time
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    all_reduce_forward_time = all_reduce_tensor[0].item(
    ) / dist.get_world_size()

    all_reduce_tensor[0] = avg_backward_time
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    all_reduce_backward_time = all_reduce_tensor[0].item(
    ) / dist.get_world_size()

    all_reduce_tensor[0] = avg_update_time
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    all_reduce_update_time = all_reduce_tensor[0].item() / dist.get_world_size(
    )

    if dist.get_rank() == 0:
        timetable = ("=====================\n"
                     "All reduce time:\n"
                     "Epoch Time(s): {:.4f}\n"
                     "Sampling Time(s): {:.4f}\n"
                     "Loading Time(s): {:.4f}\n"
                     "Forward Time(s): {:.4f}\n"
                     "Backward Time(s): {:.4f}\n"
                     "Update Time(s): {:.4f}\n"
                     "=====================".format(
                         all_reduce_epoch_time,
                         all_reduce_sample_time,
                         all_reduce_load_time,
                         all_reduce_forward_time,
                         all_reduce_backward_time,
                         all_reduce_update_time,
                     ))
        print(timetable)


def main(args):
    if args.dataset == "reddit":
        g, num_classes = load_reddit()
    elif args.dataset == "ogbn-products":
        g, num_classes = load_ogb("ogbn-products", args.root)
    elif args.dataset == "ogbn-papers100M":
        g, num_classes = load_ogb("ogbn-papers100M", args.root)
    train_nid = g.ndata["train_mask"].nonzero().flatten()
    val_nid = g.ndata["val_mask"].nonzero().flatten()
    test_nid = g.ndata["test_mask"].nonzero().flatten()
    data = train_nid, val_nid, test_nid, num_classes, g

    import torch.multiprocessing as mp
    mp.spawn(run, args=(args.num_gpus, data, args), nprocs=args.num_gpus)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")
    parser.add_argument("--n_classes",
                        type=int,
                        default=0,
                        help="the number of classes")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="the number of GPU device. Use -1 for CPU training",
    )
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_hidden", type=int, default=32)
    parser.add_argument("--heads", type=str, default="8,8,1")
    parser.add_argument("--feat_dropout", type=float, default=0.1)
    parser.add_argument("--attn_dropout", type=float, default=0.1)
    parser.add_argument("--fan_out", type=str, default="5,10,15")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--batch_size_eval", type=int, default=100000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-products",
        help="datasets: reddit, ogbn-products, ogbn-papers100M",
    )
    parser.add_argument("--root", type=str, default="/data")
    parser.add_argument("--breakdown", action="store_true")
    args = parser.parse_args()

    print(args)
    main(args)
