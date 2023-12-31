from contextlib import contextmanager

import dgl
import dgl.nn.pytorch as dglnn

import numpy as np
import torch as th
import torch.nn as nn
import tqdm

import torch.nn.functional as F
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
)


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

    def inference(self, g, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feature = g.ndata["features"]
        sampler = MultiLayerFullNeighborSampler(1)
        dataloader = DataLoader(g,
                                th.arange(g.num_nodes()).cuda(),
                                sampler,
                                device="cuda",
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=0,
                                use_uva=True)

        for l, layer in enumerate(self.layers):
            y = th.empty(g.num_nodes(),
                         self.n_hidden if l != len(self.layers) -
                         1 else self.n_classes,
                         dtype=feature.dtype)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feature[input_nodes.cpu()].cuda()
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)
                y[output_nodes] = h.to("cpu")
            feature = y
        return y


class GAT(nn.Module):

    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 n_heads,
                 activation=F.relu,
                 feat_dropout=0.6,
                 attn_dropout=0.6):
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
            layer_activation = None if i == n_layers - 1 else activation
            self.layers.append(
                dglnn.GATConv(in_dim,
                              out_dim,
                              n_heads[i],
                              feat_drop=feat_dropout,
                              attn_drop=attn_dropout,
                              activation=layer_activation,
                              allow_zero_in_degree=True))

    def forward(self, blocks, x):
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i == self.n_layers - 1:
                h = h.mean(1)
            else:
                h = h.flatten(1)
        return h

    def inference(self, g, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feature = g.ndata["features"]
        sampler = MultiLayerFullNeighborSampler(1)
        dataloader = DataLoader(g,
                                th.arange(g.num_nodes()).cuda(),
                                sampler,
                                device="cuda",
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=0,
                                use_uva=True)

        for l, layer in enumerate(self.layers):
            if l == len(self.layers) - 1:
                y = th.empty(g.num_nodes(),
                             self.n_classes * self.n_heads[l],
                             dtype=th.float32)
            else:
                y = th.empty(g.num_nodes(),
                             self.n_hidden * self.n_heads[l],
                             dtype=th.float32)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feature[input_nodes.cpu()].cuda()
                h = layer(blocks[0], x)
                if l == self.n_layers - 1:
                    h = h.mean(1)
                else:
                    h = h.flatten(1)
                y[output_nodes] = h.to("cpu")
            feature = y
        return y


class DistSAGE(nn.Module):

    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
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

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without
        neighbor sampling).

        g : the entire graph.
        x : the input of entire node set.

        Distributed layer-wise inference.
        """
        # During inference with sampling, multi-layer blocks are very
        # inefficient because lots of computations in the first few layers
        # are repeated. Therefore, we compute the representation of all nodes
        # layer by layer.  The nodes on each layer are of course splitted in
        # batches.
        # TODO: can we standardize this?
        nodes = dgl.distributed.node_split(
            np.arange(g.num_nodes()),
            g.get_partition_book(),
            force_even=True,
        )
        y = dgl.distributed.DistTensor(
            (g.num_nodes(), self.n_hidden),
            th.float32,
            "h",
            persistent=True,
        )
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                y = dgl.distributed.DistTensor(
                    (g.num_nodes(), self.n_classes),
                    th.float32,
                    "h_last",
                    persistent=True,
                )
            print(f"|V|={g.num_nodes()}, eval batch size: {batch_size}")

            sampler = dgl.dataloading.NeighborSampler([-1])
            dataloader = dgl.dataloading.DistNodeDataLoader(
                g,
                nodes,
                sampler,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)
                h = x[input_nodes].to(device)
                h_dst = h[:block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if i != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
            g.barrier()
        return y

    @contextmanager
    def join(self):
        """dummy join for standalone"""
        yield


class DistGAT(nn.Module):

    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 n_heads,
                 activation=F.relu,
                 feat_dropout=0.6,
                 attn_dropout=0.6):
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
            layer_activation = None if i == n_layers - 1 else activation
            self.layers.append(
                dglnn.GATConv(in_dim,
                              out_dim,
                              n_heads[i],
                              feat_drop=feat_dropout,
                              attn_drop=attn_dropout,
                              activation=layer_activation,
                              allow_zero_in_degree=True))

    def forward(self, blocks, x):
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i == self.n_layers - 1:
                h = h.mean(1)
            else:
                h = h.flatten(1)
        return h

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GAT model on full neighbors (i.e. without
        neighbor sampling).

        g : the entire graph.
        x : the input of entire node set.

        Distributed layer-wise inference.
        """
        nodes = dgl.distributed.node_split(
            np.arange(g.num_nodes()),
            g.get_partition_book(),
            force_even=True,
        )

        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                y = dgl.distributed.DistTensor(
                    (g.num_nodes(), self.n_classes * self.n_heads[i]),
                    th.float32,
                    "h_last",
                    persistent=True,
                )
            else:
                y = dgl.distributed.DistTensor(
                    (g.num_nodes(), self.n_hidden * self.n_heads[i]),
                    th.float32,
                    "h",
                    persistent=True,
                )
            print(f"|V|={g.num_nodes()}, eval batch size: {batch_size}")

            sampler = dgl.dataloading.NeighborSampler([-1])
            dataloader = dgl.dataloading.DistNodeDataLoader(
                g,
                nodes,
                sampler,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)
                h = x[input_nodes].to(device)
                h_dst = h[:block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if i == self.n_layers - 1:
                    h = h.mean(1)
                else:
                    h = h.flatten(1)

                y[output_nodes] = h.cpu()

            x = y
            g.barrier()
        return y

    @contextmanager
    def join(self):
        """dummy join for standalone"""
        yield


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)
