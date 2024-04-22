from time import time
import logging
import os
import os.path as osp
import numpy as np
import random
import time

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree
from torch.autograd import Variable

import random
import torch_geometric
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import StratifiedKFold
from torch import tensor
from categorizer import GraphCategorizer
from utils import stat_graph, split_class_graphs, align_graphs, align_x_graphs
from utils import split_class_x_graphs
from utils import two_graphons_mixup, two_x_graphons_mixup
from graphon_estimator import universal_svd
from models import GIN

import argparse

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d')


def add_org_nodes(data: torch_geometric.data.Data):
    data.num_nodes = torch.tensor(data.num_nodes, dtype=torch.long)
    data.org_nodes = data.num_nodes.clone().detach()
    if data.x is not None:
        data.x = data.x.to(torch.float32)
    data.edge_attr=None
    return data


def k_fold(dataset, folds, y=None):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), y):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices


def prepare_dataset_x(dataset):
    if dataset[-1].x is None or dataset[-1].x.shape[-1] == 0:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max( max_degree, degs[-1].max().item() )
            data.num_nodes = int( torch.max(data.edge_index) ) + 1

        if max_degree < 2000:
            # dataset.transform = T.OneHotDegree(max_degree)

            for data in dataset:
                degs = degree(data.edge_index[0], dtype=torch.long)
                data.x = F.one_hot(degs, num_classes=max_degree+1).to(torch.float)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            for data in dataset:
                degs = degree(data.edge_index[0], dtype=torch.long)
                data.x = ( (degs - mean) / std ).view( -1, 1 )
    return dataset



def prepare_dataset_onehot_y(dataset):

    y_set = set()
    for data in dataset:
        y_set.add(int(data.y))
    num_classes = len(y_set)

    for data in dataset:
        data.y = F.one_hot(data.y, num_classes=num_classes).to(torch.float)[0]
    return dataset


def mixup_cross_entropy_loss(input, target, size_average=True):
    """Origin: https://github.com/moskomule/mixup.pytorch
    in PyTorch's cross entropy, targets are expected to be labels
    so to predict probabilities this loss is needed
    suppose q is the target and p is the input
    loss(p, q) = -\sum_i q_i \log p_i
    """
    assert input.size() == target.size()
    assert isinstance(input, Variable) and isinstance(target, Variable)
    loss = - torch.sum(input * target)
    return loss / input.size()[0] if size_average else loss


def train(model, train_loader):
    model.train()
    loss_all = 0
    graph_all = 0
    for data in train_loader:
        # print( "data.y", data.y )
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        y = data.y.view(-1, num_classes)
        loss = mixup_cross_entropy_loss(output, y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        graph_all += data.num_graphs
        optimizer.step()
    loss = loss_all / graph_all
    return model, loss


def test(model, loader, ranges):
    model.eval()
    graph_correct = {0: 0, 1: 0, 2: 0}
    total_graphs = {0: 0, 1: 0, 2: 0}
    loss = 0
    correct = 0

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data.x, data.edge_index, data.batch)
            pred = output.max(dim=1)[1]

        y = data.y.view(-1, num_classes)
        loss += mixup_cross_entropy_loss(output, y).item() * data.num_graphs
        y = y.max(dim=1)[1]

        head_idx = torch.where((data.org_nodes >= ranges['head'][1]) & (data.org_nodes <= ranges['head'][0]), 1,
                               0).nonzero().view(-1, )
        med_idx = torch.where((data.org_nodes >= ranges['med'][1]) & (data.org_nodes <= ranges['med'][0]), 1,
                              0).nonzero().view(-1, )
        tail_idx = torch.where((data.org_nodes >= ranges['tail'][1]) & (data.org_nodes <= ranges['tail'][0]), 1,
                               0).nonzero().view(-1, )

        # print(med_idx.shape[0] + head_idx.shape[0] + tail_idx.shape[0], data.org_nodes.shape[0])
        assert med_idx.shape[0] + head_idx.shape[0] + tail_idx.shape[0] == data.org_nodes.shape[0]

        graph_correct[2] += pred[head_idx].eq(y[head_idx]).sum().item()
        graph_correct[1] += pred[med_idx].eq(y[med_idx]).sum().item()
        graph_correct[0] += pred[tail_idx].eq(y[tail_idx]).sum().item()

        total_graphs[2] += head_idx.shape[0]
        total_graphs[1] += med_idx.shape[0]
        total_graphs[0] += tail_idx.shape[0]

        # for idx, num_nodes in enumerate(data.org_nodes):
        #     if num_nodes in ranges["head"]:
        #         graph_group = 2
        #     elif num_nodes in ranges["med"]:
        #         graph_group = 1
        #     elif num_nodes in ranges["tail"]:
        #         graph_group = 0
        #     else:
        #         assert False
        #     if pred[idx].cpu().item() == data.y[idx].cpu().item():
        #         graph_correct[graph_group] += 1
        #     total_graphs[graph_group] += 1
        correct += pred.eq(y).sum().item()

    return (loss / len(loader.dataset),
            correct / len(loader.dataset),
            graph_correct[2] / total_graphs[2],
            graph_correct[1] / total_graphs[1],
            graph_correct[0] / total_graphs[0])


# def test(model, loader):
#     model.eval()
#     correct = 0
#     total = 0
#     loss = 0
#     for data in loader:
#         data = data.to(device)
#         output = model(data.x, data.edge_index, data.batch)
#         pred = output.max(dim=1)[1]
#         y = data.y.view(-1, num_classes)
#         loss += mixup_cross_entropy_loss(output, y).item() * data.num_graphs
#         y = y.max(dim=1)[1]
#         correct += pred.eq(y).sum().item()
#         total += data.num_graphs
#     acc = correct / total
#     loss = loss / total
#     return acc, loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./")
    parser.add_argument('--dataset', type=str, default="REDDIT-BINARY")
    parser.add_argument('--model', type=str, default="GIN")
    parser.add_argument('--epoch', type=int, default=800)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_hidden', type=int, default=64)
    parser.add_argument('--gmixup', type=str, default="False")
    parser.add_argument('--lam_range', type=str, default="[0.005, 0.01]")
    parser.add_argument('--aug_ratio', type=float, default=0.15)
    parser.add_argument('--aug_num', type=int, default=10)
    parser.add_argument('--gnn', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=1314)
    parser.add_argument('--log_screen', type=str, default="False")
    parser.add_argument('--ge', type=str, default="MC")
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--size_strat', action='store_true')
    parser.add_argument('--tail_aug', action='store_true')
    
    args = parser.parse_args()

    data_path = args.data_path
    dataset_name = args.dataset
    seed = args.seed
    lam_range = eval(args.lam_range)
    log_screen = eval(args.log_screen)
    gmixup = eval(args.gmixup)
    num_epochs = args.epoch

    num_hidden = args.num_hidden
    batch_size = args.batch_size
    learning_rate = args.lr
    ge = args.ge
    aug_ratio = args.aug_ratio
    aug_num = args.aug_num
    model = args.model

    if log_screen is True:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)


    logger.info('parser.prog: {}'.format(parser.prog))
    logger.info("args:{}".format(args))

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"runing device: {device}")

    path = osp.join(data_path, dataset_name)
    dataset = TUDataset(path, name=dataset_name, transform=add_org_nodes)
    total_instances = len(dataset)

    if dataset.data.x is None or dataset.data.x.shape[1] == 0:
        org_x_features = False
    else:
        org_x_features = True

    dataset = list(dataset)

    for graph in dataset:
        graph.y = graph.y.view(-1)

    val_losses, accs, head_accs, med_accs, tail_accs, durations = [], [], [], [], [], []

    nodes = list()
    for i in range(len(dataset)):
        nodes.append(dataset[i].org_nodes.item())

    split = GraphCategorizer(nodes=nodes)
    logger.info(f"NUM HEAD GRAPHS: {split.num_head_graphs}")
    logger.info(f"NUM MED GRAPHS: {split.num_med_graphs}")
    logger.info(f"NUM TAIL GRAPHS: {split.num_tail_graphs}")
    logger.info(f"SIZE RANGES: {split.size_ranges}")

    if args.size_strat:
        y = split.categories
        logger.info("[INFO] Using Size Aware Stratified Split")
    else:
        if isinstance(dataset, list):
            y = [data.y for data in dataset]
        else:
            y = dataset.data.y.tolist()

    # if not org_x_features:
    #     dataset = prepare_dataset_x(dataset)
        # org_x_features=True

    dataset = prepare_dataset_onehot_y(dataset)
    
    for fold, (train_idx, test_idx,
            val_idx) in enumerate(zip(*k_fold(dataset, args.folds, y))):

        train_dataset = [dataset[idx] for idx in train_idx.tolist()]

        avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density = stat_graph(train_dataset)
        logger.info(f"avg num nodes of training graphs: { avg_num_nodes }")
        logger.info(f"avg num edges of training graphs: { avg_num_edges }")
        logger.info(f"avg density of training graphs: { avg_density }")
        logger.info(f"median num nodes of training graphs: { median_num_nodes }")
        logger.info(f"median num edges of training graphs: { median_num_edges }")
        logger.info(f"median density of training graphs: { median_density }")

        resolution = int(median_num_nodes)

        if gmixup == True:
            if args.tail_aug:
                tail_train_dataset = list()
                for idx in train_idx:
                    if split.categories[idx] == 2:
                        tail_train_dataset.append(dataset[idx])
                logger.info("Using the Only Tail Augmentation")
            else:
                tail_train_dataset = train_dataset

            if org_x_features:
                logger.info("Using the existing features of the Dataset")
                class_graphs = split_class_x_graphs(tail_train_dataset)
                graphons = []
                for label, graphs, features in class_graphs:
                    logger.info(f"label: {label}, num_graphs:{len(graphs)}, num_features: {len(features)}")
                    align_graphs_list, align_node_features, normalized_node_degrees, max_num, min_num = align_x_graphs(
                        graphs, features, padding=True, N=resolution)
                    logger.info(f"aligned graph {align_graphs_list[0].shape}, "
                                f"align features {align_node_features[0].shape}")
                    logger.info(f"ge: {ge}")
                    graphon = universal_svd(align_graphs_list, threshold=0.2)
                    align_node_features = np.mean(align_node_features, axis=0)
                    graphons.append((label, graphon, align_node_features))

                for label, graphon, features in graphons:
                    logger.info(f"graphon info: label:{label}; mean: {graphon.mean()}, shape, {graphon.shape}, "
                                f"features: {features.shape}")
            else:
                logger.info("Creating the new features of the Dataset")
                class_graphs = split_class_graphs(tail_train_dataset)
                graphons = []
                for label, graphs in class_graphs:
                    logger.info(f"label: {label}, num_graphs:{len(graphs)}" )
                    align_graphs_list, normalized_node_degrees, max_num, min_num = align_graphs(
                        graphs, padding=True, N=resolution)
                    logger.info(f"aligned graph {align_graphs_list[0].shape}" )
                    logger.info(f"ge: {ge}")
                    graphon = universal_svd(align_graphs_list, threshold=0.2)
                    graphons.append((label, graphon))
                for label, graphon in graphons:
                    logger.info(f"graphon info: label:{label}; mean: {graphon.mean()}, shape, {graphon.shape}")

            train_nums = len(train_idx)
            num_sample = int( train_nums * aug_ratio / aug_num )
            lam_list = np.random.uniform(low=lam_range[0], high=lam_range[1], size=(aug_num,))

            new_graph = []
            mixup_func = two_x_graphons_mixup if org_x_features else two_graphons_mixup

            for lam in lam_list:
                logger.info( f"lam: {lam}" )
                logger.info(f"num_sample: {num_sample}")
                two_graphons = random.sample(graphons, 2)
                new_graph += mixup_func(two_graphons, la=lam, num_sample=num_sample)
                logger.info(f"label: {new_graph[-1].y}")

            avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density = stat_graph(new_graph)
            logger.info(f"avg num nodes of new graphs: { avg_num_nodes }")
            logger.info(f"avg num edges of new graphs: { avg_num_edges }")
            logger.info(f"avg density of new graphs: { avg_density }")
            logger.info(f"median num nodes of new graphs: { median_num_nodes }")
            logger.info(f"median num edges of new graphs: { median_num_edges }")
            logger.info(f"median density of new graphs: { median_density }")

            # train_dataset = new_graph + train_dataset

            logger.info( f"real aug ratio: {len( new_graph ) / train_nums }" )
            train_nums = train_nums + len(new_graph)

            if not org_x_features:
                temp_dataset = prepare_dataset_x(dataset + new_graph)
                train_dataset = [temp_dataset[idx] for idx in train_idx.tolist()] + temp_dataset[total_instances:]
                val_dataset = [temp_dataset[idx] for idx in val_idx.tolist()]
                test_dataset = [temp_dataset[idx] for idx in test_idx.tolist()]
            else:
                train_dataset = train_dataset + new_graph
                val_dataset = [dataset[idx] for idx in val_idx.tolist()]
                test_dataset = [dataset[idx] for idx in test_idx.tolist()]
        #     train_dataset = prepare_dataset_x(train_dataset)

        logger.info(f"num_features: {dataset[0].x.shape}" )
        logger.info(f"num_classes: {dataset[0].y.shape}"  )


        num_features = dataset[0].x.shape[-1]
        num_classes = dataset[0].y.shape[0]

        logger.info(f"train_dataset size: {len(train_dataset)}")
        logger.info(f"val_dataset size: {len(val_dataset)}")
        logger.info(f"test_dataset size: {len(test_dataset)}" )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        model = GIN(num_features=num_features, num_classes=num_classes, num_hidden=num_hidden).to(device)

        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

        for epoch in range(1, num_epochs + 1):
            model, train_loss = train(model, train_loader)
            val_loss, _, _, _, _ = test(model, val_loader, split.size_ranges)
            temp_loss, temp_acc, temp_head, temp_med, temp_tail = test(model, test_loader, split.size_ranges)
            val_losses.append(val_loss)
            accs.append(temp_acc)
            head_accs.append(temp_head)
            med_accs.append(temp_med)
            tail_accs.append(temp_tail)
            logger.info('Fold - {} Epoch: {:03d}, \nTrain Loss: {:.6f}, Val Loss: {:.6f}, Test Loss: {:.6f},  \nTest Acc: {: .6f}, Test Head Acc: {: .6f} Test Med Acc: {: .6f} Test Tail Acc: {: .6f}\n\n'.format(
                fold, epoch, train_loss, val_loss, temp_loss, temp_acc, temp_head, temp_med, temp_tail))

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    head_acc, med_acc, tail_acc = tensor(head_accs), tensor(med_accs), tensor(tail_accs)
    loss, acc = loss.view(args.folds, num_epochs), acc.view(args.folds, num_epochs)
    head_acc, med_acc, tail_acc = head_acc.view(args.folds, num_epochs), med_acc.view(args.folds, num_epochs), tail_acc.view(args.folds,
                                                                                                            num_epochs)
    loss, argmin = loss.min(dim=1)
    acc = acc[torch.arange(args.folds, dtype=torch.long), argmin]
    head_acc = head_acc[torch.arange(args.folds, dtype=torch.long), argmin]
    med_acc = med_acc[torch.arange(args.folds, dtype=torch.long), argmin]
    tail_acc = tail_acc[torch.arange(args.folds, dtype=torch.long), argmin]

    loss_mean = loss.mean().item()
    acc_mean = acc.mean().item()
    acc_std = acc.std().item()
    head_acc_mean = head_acc.mean().item()
    head_acc_std = head_acc.std().item()
    med_acc_mean = med_acc.mean().item()
    med_acc_std = med_acc.std().item()
    tail_acc_mean = tail_acc.mean().item()
    tail_acc_std = tail_acc.std().item()
    duration_mean = duration.mean().item()
    print(f'Val Loss: {loss_mean:.4f}, Test Accuracy: {acc_mean:.3f} '
            f'± {acc_std:.3f}, Duration: {duration_mean:.3f}')

    name = 'strat_metrics.txt' if args.size_strat else 'metrics.txt'
    with open(name, "a") as txt_file:
        txt_file.write(f"Dataset: {args.dataset}, \n"
                        f"Model: {args.model}, \n"
                        f"Num Layers: {args.num_hidden}, \n"
                        f"G-Mixup: {args.gmixup}, \n"
                        f"Only Tail Augmentation: {args.tail_aug}, \n"
                        f"Test Mean: {round(acc_mean, 4)}, \n"
                        f"Std Test Mean: {round(acc_std, 4)}, \n"
                        f"Head Mean: {round(head_acc_mean, 4)}, \n"
                        f"Std Head Mean: {round(head_acc_std, 4)}, \n"
                        f"Medium Mean: {round(med_acc_mean, 4)}, \n"
                        f"Std Medium Mean: {round(med_acc_std, 4)}, \n"
                        f"Tail Mean: {round(tail_acc_mean, 4)}, \n"
                        f"Std Tail Mean: {round(tail_acc_std, 4)} \n\n"
                        )
