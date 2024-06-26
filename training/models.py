from contextlib import contextmanager

import dgl
import dgl.nn.pytorch as dglnn

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.distributed as dist
import tqdm

import torch.nn.functional as F
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
)

DEFAULT_NUM_PICKS = 15


class SAGE(nn.Module):

    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        for i in range(0, n_layers):
            in_dim = in_feats if i == 0 else n_hidden
            out_dim = n_classes if i == n_layers - 1 else n_hidden
            self.layers.append(dglnn.SAGEConv(in_dim, out_dim, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


class GAT(nn.Module):

    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 n_heads,
                 activation=F.elu,
                 dropout=0.5):
        assert len(n_heads) == n_layers
        assert n_heads[-1] == 1

        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_heads = n_heads

        self.layers = nn.ModuleList()
        for i in range(0, n_layers):
            in_dim = in_feats if i == 0 else n_hidden * n_heads[i - 1]
            out_dim = n_classes if i == n_layers - 1 else n_hidden
            if i == 0:
                self.layers.append(
                    dglnn.SAGEConv(in_dim, out_dim * n_heads[i], "mean"))
            else:
                self.layers.append(
                    dglnn.GATConv(in_dim,
                                  out_dim,
                                  n_heads[i],
                                  allow_zero_in_degree=True))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i == self.n_layers - 1:
                h = h.mean(1)
            else:
                h = self.activation(h)
                h = self.dropout(h)
                h = h.flatten(1)
        return h


def nodewise_inference(model, dataloader, labels, device="cuda"):
    model.eval()
    with torch.no_grad():
        acc = 0
        length = 0
        for input_nodes, output_nodes, blocks in tqdm.tqdm(
                dataloader, disable=dist.get_rank() != 0):
            input_nodes = input_nodes.to(device)
            output_nodes = output_nodes.to(device)
            blocks = [block.to(device) for block in blocks]
            batch_inputs = blocks[0].srcdata["features"]
            pred = model(blocks, batch_inputs).cpu()
            output_labels = labels[output_nodes.cpu()]
            acc += (torch.argmax(
                pred, dim=1).cpu() == output_labels.cpu()).float().sum()
            length += output_nodes.numel()
        return acc / length


class DistSAGE(nn.Module):

    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        for i in range(0, n_layers):
            in_dim = in_feats if i == 0 else n_hidden
            out_dim = n_classes if i == n_layers - 1 else n_hidden
            self.layers.append(dglnn.SAGEConv(in_dim, out_dim, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    # def inference(self, g, x, batch_size, device):
    #     """
    #     Inference with the GraphSAGE model on full neighbors (i.e. without
    #     neighbor sampling).

    #     g : the entire graph.
    #     x : the input of entire node set.

    #     Distributed layer-wise inference.
    #     """
    #     # During inference with sampling, multi-layer blocks are very
    #     # inefficient because lots of computations in the first few layers
    #     # are repeated. Therefore, we compute the representation of all nodes
    #     # layer by layer.  The nodes on each layer are of course splitted in
    #     # batches.
    #     # TODO: can we standardize this?
    #     nodes = dgl.distributed.node_split(
    #         np.arange(g.num_nodes()),
    #         g.get_partition_book(),
    #         force_even=True,
    #     )
    #     y = dgl.distributed.DistTensor(
    #         (g.num_nodes(), self.n_hidden),
    #         th.float32,
    #         "h",
    #         persistent=True,
    #     )
    #     for i, layer in enumerate(self.layers):
    #         if i == len(self.layers) - 1:
    #             y = dgl.distributed.DistTensor(
    #                 (g.num_nodes(), self.n_classes),
    #                 th.float32,
    #                 "h_last",
    #                 persistent=True,
    #             )
    #         print(f"|V|={g.num_nodes()}, eval batch size: {batch_size}")

    #         sampler = dgl.dataloading.NeighborSampler([-1])
    #         dataloader = dgl.dataloading.DistNodeDataLoader(
    #             g,
    #             nodes,
    #             sampler,
    #             batch_size=batch_size,
    #             shuffle=False,
    #             drop_last=False,
    #         )

    #         for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
    #             block = blocks[0].to(device)
    #             h = x[input_nodes].to(device)
    #             h_dst = h[:block.number_of_dst_nodes()]
    #             h = layer(block, (h, h_dst))
    #             if i != len(self.layers) - 1:
    #                 h = self.activation(h)
    #                 h = self.dropout(h)

    #             y[output_nodes] = h.cpu()

    #         x = y
    #         g.barrier()
    #     return y

    # @contextmanager
    # def join(self):
    #     """dummy join for standalone"""
    #     yield


class DistGAT(nn.Module):

    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 n_heads,
                 activation=F.elu,
                 dropout=0.5):
        assert len(n_heads) == n_layers
        assert n_heads[-1] == 1

        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_heads = n_heads

        self.layers = nn.ModuleList()
        for i in range(0, n_layers):
            in_dim = in_feats if i == 0 else n_hidden * n_heads[i - 1]
            out_dim = n_classes if i == n_layers - 1 else n_hidden
            if i == 0:
                self.layers.append(
                    dglnn.SAGEConv(in_dim, out_dim * n_heads[i], "mean"))
            else:
                self.layers.append(
                    dglnn.GATConv(in_dim,
                                  out_dim,
                                  n_heads[i],
                                  allow_zero_in_degree=True))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i == self.n_layers - 1:
                h = h.mean(1)
            else:
                h = self.activation(h)
                h = self.dropout(h)
                h = h.flatten(1)
        return h

    # def inference(self, g, x, batch_size, device):
    #     """
    #     Inference with the GAT model on full neighbors (i.e. without
    #     neighbor sampling).

    #     g : the entire graph.
    #     x : the input of entire node set.

    #     Distributed layer-wise inference.
    #     """
    #     nodes = dgl.distributed.node_split(
    #         np.arange(g.num_nodes()),
    #         g.get_partition_book(),
    #         force_even=True,
    #     )

    #     for i, layer in enumerate(self.layers):
    #         if i == len(self.layers) - 1:
    #             y = dgl.distributed.DistTensor(
    #                 (g.num_nodes(), self.n_classes * self.n_heads[i]),
    #                 th.float32,
    #                 "h_last",
    #                 persistent=True,
    #             )
    #         else:
    #             y = dgl.distributed.DistTensor(
    #                 (g.num_nodes(), self.n_hidden * self.n_heads[i]),
    #                 th.float32,
    #                 "h",
    #                 persistent=True,
    #             )
    #         print(f"|V|={g.num_nodes()}, eval batch size: {batch_size}")

    #         sampler = dgl.dataloading.NeighborSampler([-1])
    #         dataloader = dgl.dataloading.DistNodeDataLoader(
    #             g,
    #             nodes,
    #             sampler,
    #             batch_size=batch_size,
    #             shuffle=False,
    #             drop_last=False,
    #         )

    #         for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
    #             block = blocks[0].to(device)
    #             h = x[input_nodes].to(device)
    #             h_dst = h[:block.number_of_dst_nodes()]
    #             h = layer(block, (h, h_dst))
    #             if i == self.n_layers - 1:
    #                 h = h.mean(1)
    #             else:
    #                 h = h.flatten(1)

    #             y[output_nodes] = h.cpu()

    #         x = y
    #         g.barrier()
    #     return y

    # @contextmanager
    # def join(self):
    #     """dummy join for standalone"""
    #     yield


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)
