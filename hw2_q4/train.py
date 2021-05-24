
import argparse
import time

import networkx as nx
import numpy as np
import torch
from torch._C import get_num_interop_threads
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

import torch_geometric.nn as pyg_nn

import models
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def arg_parse():
    parser = argparse.ArgumentParser(description='GNN arguments.')
    utils.parse_optimizer(parser)

    parser.add_argument('--model_type', type=str,
                        help='Type of GNN model.')
    parser.add_argument('--batch_size', type=int,
                        help='Training batch size')
    parser.add_argument('--num_layers', type=int,
                        help='Number of graph conv layers')
    parser.add_argument('--hidden_dim', type=int,
                        help='Training hidden size')
    parser.add_argument('--dropout', type=float,
                        help='Dropout rate')
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs')
    parser.add_argument('--dataset', type=str,
                        help='Dataset')

    parser.set_defaults(model_type='GCN',
                        dataset='cora',
                        num_layers=2,
                        batch_size=32,
                        hidden_dim=16,
                        dropout=0.5,
                        epochs=200,
                        opt='adam',   # opt_parser
                        opt_scheduler='none',
                        weight_decay=0,
                        lr=0.01,
                        num_heads=8,
                        concat=True,
                        )

    return parser.parse_args()

def train(dataset, task, args):
    if task == 'graph':
        # graph classification: separate dataloader for test set
        data_size = len(dataset)
        loader = DataLoader(
                dataset[:int(data_size * 0.7)], batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[int(data_size * 0.7):int(data_size*0.8)], batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(
                dataset[int(data_size * 0.8):], batch_size=args.batch_size, shuffle=False)
    elif task == 'node':
        # use mask to split train/validation/test
        valid_loader = test_loader = loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        raise RuntimeError('Unknown task')

    # build model
    model = models.GNNStack(dataset.num_node_features, args.hidden_dim, dataset.num_classes, 
                            args, task=task)
    print(model)
    model.to(device)
    scheduler, opt = utils.build_optimizer(args, model.parameters())

    # train
    trains, vals = [], []
    best_val_acc, stop_cnt, early_stop = 0, 0, 1000
    best_test_acc = 0
    for epoch in range(args.epochs):
        total_loss = 0
        train_acc = 0
        model.train()
        for batch in loader: # every time run on the total dataset
            batch.to(device)
            opt.zero_grad()
            pred = model(batch)
            label = batch.y
            if task == 'node':
                pred = pred[batch.train_mask]
                label = label[batch.train_mask]
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
            train_acc += pred.max(dim=1)[1].eq(label).sum().item()
        total_loss /= len(loader.dataset)
        # print(total_loss)
        ### diff from origin
        if model.task == 'node':
            for data in loader.dataset:
                train_acc /= torch.sum(data.train_mask).item()
        else:
            train_acc /= len(loader.dataset)
        
        trains.append(train_acc)
        val_acc, valid_samples = test(valid_loader, model, is_validation=True)
        vals.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test(test_loader, model)
            stop_cnt = 0
        else:
            stop_cnt += 1

        print("Loss in Epoch {:03d}: {:.4f}. ".format(epoch, total_loss), end="")
        print("Current Best Val Acc {:.4f} with {:04d} samples, with Train Acc {:.4f}".format(best_val_acc, valid_samples, train_acc))

        if stop_cnt >= early_stop:
            break
        ### diff from origin

        if epoch % 10 == 0:
            test_acc, test_samples = test(test_loader, model)
            print("[Epoch {:03d}] Current Best Test Acc {:.4f} with {:04d} samples".format(epoch, test_acc, test_samples))
    
    return list(range(1, args.epochs+1)), vals, trains, best_test_acc


def test(loader, model, is_validation=False):
    model.eval()

    correct = 0
    for data in loader:
        data.to(device)
        with torch.no_grad():
            # max(dim=1) returns values, indices tuple; only need indices
            pred = model(data).max(dim=1)[1]
            label = data.y

        if model.task == 'node':
            mask = data.val_mask if is_validation else data.test_mask
            # node classification: only evaluate on nodes in test set
            pred = pred[mask]
            label = data.y[mask]
            
        correct += pred.eq(label).sum().item()
    
    if model.task == 'graph':
        total = len(loader.dataset) 
    else:
        total = 0
        for data in loader.dataset:
            total += torch.sum(data.test_mask).item() if not is_validation else torch.sum(data.val_mask).item()
        # print("total", total)
    return correct / total, total

from matplotlib import pyplot as plt
def main():
    args = arg_parse()

    args.dataset='cora'

    if args.dataset == 'enzymes':
        dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
        task = 'graph'
        print("# graphs, %d"%len(dataset))
    elif args.dataset == 'cora':
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
        task = 'node'
        print("# nodes, %d"%dataset.data.num_nodes)
    
    gcn_epoch, gcn_vals, gcn_trains, gcn_best_test = train(dataset, task, args) 
    plt.plot(gcn_epoch, gcn_vals, label="GCN-val")
    plt.plot(gcn_epoch, gcn_trains, linestyle='--', label="GCN-train")

    args.model_type = "GraphSage"
    args.hidden_dim = 256
    gcn_epoch, gcn_vals, gcn_trains, graphsage_best_test = train(dataset, task, args) 
    plt.plot(gcn_epoch, gcn_vals, label="GraphSage-val")
    plt.plot(gcn_epoch, gcn_trains, linestyle='--', label="GraphSage-train")

    args.model_type = "GAT"
    args.hidden_dim = 16
    gcn_epoch, gcn_vals, gcn_trains, gat_best_test = train(dataset, task, args) 
    plt.plot(gcn_epoch, gcn_vals, label="GAT-val")
    plt.plot(gcn_epoch, gcn_trains, linestyle='--', label="GAT-train")

    plt.title("Validation Accuracy Changes over Epochs on %s Dataset"%(args.dataset))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()
    print(gcn_best_test, graphsage_best_test, gat_best_test)

if __name__ == '__main__':
    main()


