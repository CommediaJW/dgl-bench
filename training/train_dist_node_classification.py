import argparse
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
from models import SAGE, GAT, compute_acc

torch.manual_seed(25)


def feature_initializer(shape, dtype):
    size = []
    for s in shape:
        size.append(s)
    tensor = torch.randn(shape, dtype=dtype)
    return tensor


def load_subtensor(g, seeds, input_nodes, device, load_feat=True):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = (g.ndata["features"][input_nodes].float().to(device)
                    if load_feat else None)
    batch_labels = g.ndata["labels"][seeds].to(device)
    return batch_inputs, batch_labels


def evaluate(model, g, inputs, labels, val_nid, test_nid, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with torch.no_grad():
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred[val_nid],
                       labels[val_nid]), compute_acc(pred[test_nid],
                                                     labels[test_nid])


def run(args, device, data):
    # Unpack data
    train_nid, val_nid, test_nid, n_classes, g = data
    # prefetch_node_feats/prefetch_labels are not supported for DistGraph yet.
    fan_out = [int(fanout) for fanout in args.fan_out.split(",")]
    sampler = dgl.dataloading.NeighborSampler(fan_out)
    dataloader = dgl.dataloading.DistNodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    # Define model and optimizer
    if args.model == "sage":
        model = SAGE(g.ndata["features"].shape[1], args.num_hidden, n_classes,
                     len(fan_out), F.relu, args.dropout)
    elif args.model == "gat":
        heads = [args.num_heads for _ in range(len(fan_out) - 1)]
        heads.append(1)
        num_hidden = args.num_hidden // args.num_heads
        model = GAT(g.ndata["features"].shape[1], num_hidden, n_classes,
                    len(fan_out), heads, F.elu, args.dropout)
    model = model.to(device)
    if not args.standalone:
        if args.num_trainers == -1:
            model = torch.nn.parallel.DistributedDataParallel(model)
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[device], output_device=device)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.978)

    pb = g.get_partition_book()
    partition_range = [0]
    for i in range(pb.num_partitions()):
        partition_range.append(pb._max_node_ids[i])
    part_id = pb.partid

    epoch_time_log = []
    sample_time_log = []
    load_time_log = []
    forward_time_log = []
    backward_time_log = []
    update_time_log = []
    num_layer_seeds_log = []
    num_layer_neighbors_log = []
    num_inputs_log = []
    for epoch in range(args.num_epochs):

        sample_time = 0
        load_time = 0
        forward_time = 0
        backward_time = 0
        update_time = 0
        num_seeds = 0
        num_inputs = 0
        num_iters = 0
        num_layer_seeds = 0
        num_layer_neighbors = 0

        model.train()
        with model.join():
            if args.breakdown:
                dist.barrier()
                torch.cuda.synchronize()
            epoch_tic = time.time()
            sample_begin = time.time()
            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                for l, block in enumerate(blocks):
                    layer_seeds = block.dstdata[dgl.NID]
                    remote_layer_seeds_num = torch.sum(
                        layer_seeds >= partition_range[
                            part_id + 1]).item() + torch.sum(
                                layer_seeds < partition_range[part_id]).item()
                    num_layer_seeds += remote_layer_seeds_num
                    num_layer_neighbors += remote_layer_seeds_num * fan_out[l]
                blocks = [block.to(device) for block in blocks]
                num_iters += 1
                if args.breakdown:
                    torch.cuda.synchronize()
                sample_time += time.time() - sample_begin

                load_begin = time.time()
                batch_inputs = g.ndata["features"][input_nodes].float().to(
                    device)
                batch_labels = g.ndata["labels"][seeds].to(device).long()
                num_seeds += len(blocks[-1].dstdata[dgl.NID])
                num_inputs += torch.sum(input_nodes >= partition_range[
                    part_id + 1]).item() + torch.sum(
                        input_nodes < partition_range[part_id]).item()
                if args.breakdown:
                    torch.cuda.synchronize()
                load_time += time.time() - load_begin

                forward_start = time.time()
                batch_pred = model(blocks, batch_inputs)
                loss = loss_fcn(batch_pred, batch_labels)
                if args.breakdown:
                    torch.cuda.synchronize()
                forward_time += time.time() - forward_start

                backward_begin = time.time()
                optimizer.zero_grad()
                loss.backward()
                if args.breakdown:
                    torch.cuda.synchronize()
                backward_time += time.time() - backward_begin

                update_start = time.time()
                optimizer.step()
                if args.breakdown:
                    torch.cuda.synchronize()
                update_time += time.time() - update_start

                if (step + 1) % args.log_every == 0:
                    acc = compute_acc(batch_pred, batch_labels)
                    gpu_mem_alloc = (torch.cuda.max_memory_allocated() /
                                     1000000
                                     if torch.cuda.is_available() else 0)
                    print(
                        "Part {} | Epoch {:05d} | Step {:05d} | Loss {:.4f} | "
                        "Train Acc {:.4f} | GPU {:.1f} MB".format(
                            dist.get_rank(), epoch, step, loss.item(),
                            acc.item(), gpu_mem_alloc))
                    train_acc_tensor = torch.tensor([acc.item()]).cuda()
                    dist.all_reduce(train_acc_tensor, dist.ReduceOp.SUM)
                    train_acc_tensor /= dist.get_world_size()
                    if dist.get_rank() == 0:
                        print("Avg train acc {:.4f}".format(
                            train_acc_tensor[0].item()))

                if args.breakdown:
                    torch.cuda.synchronize()
                sample_begin = time.time()
            scheduler.step()

            epoch_toc = time.time()

        for i in range(args.num_trainers):
            dist.barrier()
            if i == dist.get_rank() % args.num_trainers:
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
                             "#sampling_seeds: {}\n"
                             "#sampled_neighbors: {}\n"
                             "=====================".format(
                                 dist.get_rank(),
                                 epoch_toc - epoch_tic,
                                 sample_time,
                                 load_time,
                                 forward_time,
                                 backward_time,
                                 update_time,
                                 num_seeds,
                                 num_inputs,
                                 num_iters,
                                 num_layer_seeds,
                                 num_layer_neighbors,
                             ))
                print(timetable)
        sample_time_log.append(sample_time)
        load_time_log.append(load_time)
        forward_time_log.append(forward_time)
        backward_time_log.append(backward_time)
        update_time_log.append(update_time)
        epoch_time_log.append(epoch_toc - epoch_tic)
        num_layer_seeds_log.append(num_layer_seeds)
        num_layer_neighbors_log.append(num_layer_neighbors)
        num_inputs_log.append(num_inputs)

        # if (epoch + 1) % args.eval_every == 0 and epoch != 0:
        #     start = time.time()
        #     val_acc, test_acc = evaluate(
        #         model if args.standalone else model.module,
        #         g,
        #         g.ndata["features"],
        #         g.ndata["labels"],
        #         val_nid,
        #         test_nid,
        #         args.batch_size_eval,
        #         device,
        #     )
        #     print("Part {}, Val Acc {:.4f}, Test Acc {:.4f}, time: {:.4f}".
        #           format(dist.get_rank(), val_acc, test_acc,
        #                  time.time() - start))
        #     acc_tensor = torch.tensor([val_acc, test_acc]).cuda()
        #     dist.all_reduce(acc_tensor, dist.ReduceOp.SUM)
        #     acc_tensor /= dist.get_world_size()
        #     if dist.get_rank() == 0:
        #         print("All parts avg val acc {:.4f}, test acc {:.4f}".format(
        #             acc_tensor[0].item(), acc_tensor[1].item()))

    avg_epoch_time = np.mean(epoch_time_log[1:])
    avg_sample_time = np.mean(sample_time_log[1:])
    avg_load_time = np.mean(load_time_log[1:])
    avg_forward_time = np.mean(forward_time_log[1:])
    avg_backward_time = np.mean(backward_time_log[1:])
    avg_update_time = np.mean(update_time_log[1:])
    avg_remote_inputs = np.mean(num_inputs_log[1:])
    avg_remote_seeds = np.mean(num_layer_seeds_log[1:])
    avg_remote_neighbors = np.mean(num_layer_neighbors_log[1:])

    for i in range(args.num_trainers):
        dist.barrier()
        if i == dist.get_rank() % args.num_trainers:
            timetable = ("=====================\n"
                         "Part {}, Avg Time:\n"
                         "Epoch Time(s): {:.4f}\n"
                         "Sampling Time(s): {:.4f}\n"
                         "Loading Time(s): {:.4f}\n"
                         "Forward Time(s): {:.4f}\n"
                         "Backward Time(s): {:.4f}\n"
                         "Update Time(s): {:.4f}\n"
                         "#inputs: {}\n"
                         "#sampling_seeds: {}\n"
                         "#sampled_neighbors: {}\n"
                         "=====================".format(
                             dist.get_rank(), avg_epoch_time, avg_sample_time,
                             avg_load_time, avg_forward_time,
                             avg_backward_time, avg_update_time,
                             avg_remote_inputs, avg_remote_seeds,
                             avg_remote_neighbors))
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

    all_reduce_tensor[0] = train_nid.shape[0]
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    train_total_num = all_reduce_tensor[0].item()

    all_reduce_tensor[0] = avg_remote_inputs
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    avg_remote_inputs = all_reduce_tensor[0].item()

    all_reduce_tensor[0] = avg_remote_seeds
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    avg_remote_seeds = all_reduce_tensor[0].item()

    all_reduce_tensor[0] = avg_remote_neighbors
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    avg_remote_neighbors = all_reduce_tensor[0].item()

    if dist.get_rank() == 0:
        timetable = ("=====================\n"
                     "All reduce time:\n"
                     "Throughput(seeds/sec): {:.4f}\n"
                     "Epoch Time(s): {:.4f}\n"
                     "Sampling Time(s): {:.4f}\n"
                     "Loading Time(s): {:.4f}\n"
                     "Forward Time(s): {:.4f}\n"
                     "Backward Time(s): {:.4f}\n"
                     "Update Time(s): {:.4f}\n"
                     "#Remote Loading Nodes: {:.4f}\n"
                     "#Remote Seeds: {:.4f}\n"
                     "#Remote Neighbors: {:.4f}\n"
                     "=====================".format(
                         train_total_num / all_reduce_epoch_time,
                         all_reduce_epoch_time, all_reduce_sample_time,
                         all_reduce_load_time, all_reduce_forward_time,
                         all_reduce_backward_time, all_reduce_update_time,
                         avg_remote_inputs, avg_remote_seeds,
                         avg_remote_neighbors))
        print(timetable)


def main(args):
    dgl.distributed.initialize(args.ip_config)
    if not args.standalone:
        dist.init_process_group(backend=args.backend)

    g = dgl.distributed.DistGraph(args.graph_name,
                                  part_config=args.part_config)
    print("rank:", dist.get_rank())

    pb = g.get_partition_book()
    print(g.ndata)

    train_nid = dgl.distributed.node_split(g.ndata["train_mask"],
                                           pb,
                                           force_even=True)

    val_nid = dgl.distributed.node_split(g.ndata["val_mask"],
                                         pb,
                                         force_even=True)
    test_nid = dgl.distributed.node_split(g.ndata["test_mask"],
                                          pb,
                                          force_even=True)
    if "features" not in g.ndata:
        print("Generate features...")
        if args.graph_name == "ogb-products":
            feature_dim = 100
            feature_dtype = torch.float32
        elif args.graph_name == "ogb-paper100M":
            feature_dim = 128
            feature_dtype = torch.float32
        elif args.graph_name == "mag240m":
            feature_dim = 768
            feature_dtype = torch.float16
        features = dgl.distributed.DistTensor((g.num_nodes(), feature_dim),
                                              feature_dtype,
                                              "features",
                                              attach=False,
                                              init_func=feature_initializer)
        dist.barrier()
        g.ndata["features"] = features

    local_nid = pb.partid2nids(pb.partid).detach().numpy()
    print("part {}, train: {} (local: {}), val: {} (local: {}), test: {} "
          "(local: {})".format(
              dist.get_rank(),
              len(train_nid),
              len(np.intersect1d(train_nid.numpy(), local_nid)),
              len(val_nid),
              len(np.intersect1d(val_nid.numpy(), local_nid)),
              len(test_nid),
              len(np.intersect1d(test_nid.numpy(), local_nid)),
          ))
    dev_id = dist.get_rank() % args.num_trainers
    device = torch.device("cuda:" + str(dev_id))
    torch.cuda.set_device(device)
    labels = g.ndata["labels"][np.arange(g.num_nodes())]
    n_classes = int(torch.max(labels[~torch.isnan(labels)]).item() + 1)
    print("num_classes: {}".format(n_classes))

    # Pack data
    data = train_nid, val_nid, test_nid, n_classes, g
    run(args, device, data)
    print("parent ends")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")
    parser.add_argument("--graph_name", type=str, help="graph name")
    parser.add_argument("--id", type=int, help="the partition id")
    parser.add_argument("--ip_config",
                        type=str,
                        help="The file for IP configuration")
    parser.add_argument("--part_config",
                        type=str,
                        help="The path to the partition config file")
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        help="pytorch distributed backend",
    )
    parser.add_argument(
        "--num_trainers",
        type=int,
        default=8,
        help="the number of GPU device. Use -1 for CPU training",
    )
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--model",
                        type=str,
                        default="sage",
                        choices=["sage", "gat"])
    parser.add_argument("--num_hidden", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--fan_out", type=str, default="20,20,20")
    parser.add_argument("--batch_size", type=int, default=1536)
    parser.add_argument("--batch_size_eval", type=int, default=100000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=999)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--local_rank",
                        type=int,
                        help="get rank of the process")
    parser.add_argument("--standalone",
                        action="store_true",
                        help="run in the standalone mode")
    parser.add_argument("--breakdown", action="store_true")
    args = parser.parse_args()

    print(args)
    main(args)
