#!/usr/bin/env python
# coding: utf-8

# train the GNN model using mincut_pool loss
# save the embeddings right before softmax out


import os
import sys

import torch
from torch_geometric.data import Dataset, Data, download_url
from torch_geometric.loader import DataLoader

import numpy as np
import pandas as pd
import json
import pickle
import matplotlib
import matplotlib.path as mplPath
import matplotlib.pyplot as plt
import networkx as nx
import warnings

from torch_geometric.utils import remove_self_loops

from tqdm import tqdm

from scipy.spatial import Delaunay
from scipy.spatial import distance_matrix

from sklearn import metrics

from collections.abc import Sequence
from collections import Counter
from collections import defaultdict
import copy

import random
import math

import gc

import argparse

# this line below is needed if want to import function from other python files in local directory
sys.path.append(os.getcwd())

from data_utilities import data_features, make_model_run_id
from models import GCN_model

from data_transformers import add_num_of_cells
from sparse_mincut_pool import sparse_mincut_pool
from graph_data_class import CellularGraphDataset


parser = argparse.ArgumentParser(description='classification on cords 2024 dataset')

parser.add_argument('--data_name', default="cords_2024", type=str, help='the name of the dataset to use')
parser.add_argument('--graph_type', default="extended", type=str,
                                    help='name for the feature and graph combination.',
                                    choices=['extended', 'basic', 'local'])
parser.add_argument('--gcn_type', default="gat2", type=str,
                                   choices=['gcn', 'gat', 'gat2'],
                                   help='what gcn layer to use')
parser.add_argument('--skip_type', default="add", type=str,
                                   choices=['no', 'add', 'add2'],
                                   help='which type of skip connection to use')
parser.add_argument('--device', default="gpu", type=str, help='whether to use CPU or GPU')
parser.add_argument('--s_dim2', default=40, type=int, help='number of columns in soft assignment matrix')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate, for example, 0.0001')
parser.add_argument('--epoch_limit', type=int, default=1000, help='max number of epoches, for example, 1000')
parser.add_argument('--degree_limit', type=int, default=20, help='the average degree to achieve for each image in fully extended graph, for example, 20')
parser.add_argument('--data_dir', default="./data/Cords_data", type=str, help='the overall folder of the dataset')
parser.add_argument('--dist_cutoff', type=float, default=16, help='distance cutoff for constructing the base graph')
parser.add_argument('--expanded_edge_cutoff', type=float, default=48, help='search radius for candidate cells in the first extension step')
parser.add_argument('--top_k', type=int, default=4, help='number of nearest candidate cells to add in the first extension step')
parser.add_argument('--ctg_comp_dist_cutoff', type=float, default=0.176, help='cell type group composition distance cutoff for adding edges')
parser.add_argument('--group_scheme', type=str, default="default", help='fine-to-coarse cell type grouping scheme')
parser.add_argument('--model_run_id', type=str, default=None, help='dataset-aware identifier for this graph/model parameter set')
parser.add_argument('--random_seed_index', type=int, default=1, help='which triple of random seeds to use, counting from 1: '
'the seeds at positions [3*(index-1), 3*(index-1)+1, 3*(index-1)+2] of the random seed file are used for torch, random '
'and numpy respectively. random_seeds_40_triples.txt holds 120 seeds, so this takes values 1-40')
parser.add_argument('--random_seed_file', type=str, default="./data/random_seeds_40_triples.txt",
                    help='text file with one random seed per line')


def mincutpool_run(data_name="cords_2024", graph_type="extended", gcn_type="gat2", skip_type="add",
                   device="gpu", s_dim2=40, batch_size=64,
                   lr=0.0001, epoch_limit=1000, degree_limit=20,
                   data_dir="./data/Cords_data", dist_cutoff=16,
                   expanded_edge_cutoff=48, top_k=4,
                   ctg_comp_dist_cutoff=0.176, group_scheme="default",
                   model_run_id=None,
                   random_seed_index=1,
                   random_seed_file="./data/random_seeds_40_triples.txt"):

    # the random seeds are read from a file rather than hard-coded, so that the
    # same code can be run repeatedly with a different, non-overlapping seed
    # triple each time. random_seed_index counts from 1
    with open(random_seed_file) as f:
        all_random_seeds = [int(line) for line in f if line.strip() != ""]

    seed_positions = [3 * (random_seed_index - 1),
                      3 * (random_seed_index - 1) + 1,
                      3 * (random_seed_index - 1) + 2]

    if random_seed_index < 1 or seed_positions[2] >= len(all_random_seeds):
        raise ValueError("random_seed_index " + str(random_seed_index) + " needs positions " +
                         str(seed_positions) + ", but " + random_seed_file + " only has " +
                         str(len(all_random_seeds)) + " seeds")

    torch_seed, random_seed, numpy_seed = [all_random_seeds[p] for p in seed_positions]

    print("random seed file:", random_seed_file)
    print("random seed index:", random_seed_index, "-> positions", seed_positions)
    print("torch.manual_seed(" + str(torch_seed) + "),",
          "random.seed(" + str(random_seed) + "),",
          "np.random.seed(" + str(numpy_seed) + ")")

    torch.manual_seed(torch_seed)
    random.seed(random_seed)
    np.random.seed(numpy_seed)

    input_args = locals()
    print("input args are", input_args)

    cell_feature = "comp2nd"
    mincut_type = "sparse_mincut_pool"
    n_gcns = 2

    # Metadata for the chosen dataset
    # not split on dataset based on patients into training/validation/test
    # all images go to training set
    if model_run_id is None:
        model_run_id = make_model_run_id(data_name, group_scheme, dist_cutoff,
                                         expanded_edge_cutoff, top_k,
                                         ctg_comp_dist_cutoff, degree_limit)

    data_dicts  = data_features(data_name, graph_type,
                                group_scheme=group_scheme,
                                dist_cutoff=dist_cutoff,
                                expanded_edge_cutoff=expanded_edge_cutoff,
                                top_k=top_k,
                                ctg_comp_dist_cutoff=ctg_comp_dist_cutoff,
                                degree_limit=degree_limit,
                                model_run_id=model_run_id,
                                data_dir=data_dir)
    
    NEIGHBOR_EDGE_CUTOFF = data_dicts.dist_cutoff
    PATH_PURITY_CUTOFF = data_dicts.path_purity_cutoff

    result_subfolder = data_dicts.result_subfolder

    # model_run_id itself is left unsuffixed, because it is what locates the
    # graph objects to load. The seed index is appended only to the output
    # paths, so that the repeated runs do not overwrite each other
    result_subfolder_rs = result_subfolder+"__repeat_"+str(random_seed_index)

    output_dir = "./results_rs/"+result_subfolder_rs+"/"+graph_type
    model_dir = "./saved_models_rs/"+result_subfolder_rs+"/"+graph_type

    embedding_save_dir = output_dir+"/epoch_"+str(epoch_limit)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    os.makedirs(embedding_save_dir, exist_ok=True)

    # Settings
    raw_data_root = data_dicts.raw_dir
    dataset_root = data_dicts.dataset_root

    if device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)

    train_images = data_dicts.train_images

    print("The number of images involved: ")
    print(len(train_images))


    # Define Cellular Graph Dataset
    dataset_kwargs = {
        'raw_cell_info_path': raw_data_root,
        'raw_folder_name': 'graph',
        'processed_folder_name': data_dicts.processed_folder_name,
        'node_features': ["cell_type_group", "neighborhood_composition"],
        'neighbor_edge_cutoff': NEIGHBOR_EDGE_CUTOFF,
        'expanded_edge_cutoff': expanded_edge_cutoff,
        'top_k': top_k,
        'degree_limit': degree_limit,
        'ctg_comp_dist_cutoff': ctg_comp_dist_cutoff,
        'path_purity_cutoff': PATH_PURITY_CUTOFF,
        'cell_type_mapping': data_dicts.cell_type_mapping,
        'group_ct_mapping': data_dicts.group_ct_mapping,
        'operation_type': "load"
    }


    dataset_kwargs.keys()

    dataset_root

    dataset = CellularGraphDataset(dataset_root, **dataset_kwargs)

    N_CELL_TYPE_GROUPS = len(data_dicts.group_ct_mapping)

    # Define Transformers
    transformers = [
        add_num_of_cells()
    ]

    dataset.set_transforms(transformers)

    len(dataset)
    len(dataset.region_ids)
    len(dataset.raw_paths)

    dataset
    dataset.raw_paths[-1]

    # get data objects corresponding to training images
    region_ids = [dataset.get_full(i).region_id for i in range(dataset.N)]

    train_dataset = dataset.index_select([i for i,x in enumerate(region_ids) if x in train_images])

    train_dataset

    len(train_dataset.processed_paths)
    max(train_dataset.indices())


    assert len(set(train_dataset.region_ids))==len(train_images), "numbers of images do not align"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    images_loader = DataLoader(train_dataset, batch_size=1, shuffle=False) # for saving the embeddings for each image separately

    model = GCN_model(cell_feature, N_CELL_TYPE_GROUPS,
                      gcn_type, n_gcns, s_dim2, mincut_type, skip_type)
    print(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    def train(mincut_type):
        model.zero_grad(set_to_none=True)
        # model.to(device)
        model.train()

        for data in train_loader:  # Iterate in batches over the training dataset.
            # print(data)
            data = data.to(device)
            if mincut_type == "sparse_mincut_pool":
                mc_loss, o1_loss = model(data.x,
                                         remove_self_loops(data.edge_index)[0],
                                         data.batch,
                                         data.n_cells)   # Perform a single forward pass.
                loss = mc_loss + o1_loss # Compute the loss.

            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            model.zero_grad(set_to_none=True)  # Clear gradients.


    def record_loss(loader, mincut_type):

        model.to(device)
        model.eval()

        mc_loss_list = []
        o_loss_list = []

        with torch.no_grad():
            for data in loader:  # Iterate in batches over the training/validation/test dataset.

                data = data.to(device)

                if mincut_type == "sparse_mincut_pool":

                    mc_loss, o_loss = model(data.x,
                                             remove_self_loops(data.edge_index)[0],
                                             data.batch,
                                             data.n_cells)

                    mc_loss_value = mc_loss.to('cpu').item()
                    o_loss_value = o_loss.to('cpu').item()

                    mc_loss_list += [mc_loss_value]
                    o_loss_list += [o_loss_value]


        mc_ave = sum(mc_loss_list)/len(mc_loss_list)
        o_ave = sum(o_loss_list)/len(o_loss_list)

        return mc_ave, o_ave

    if device != "cpu":
        print("before training: ")
        print(f"gpu used {torch.cuda.max_memory_allocated(device=None)} memory")


    train_mc_list = []
    train_o_list = []
    train_unsuper_list = []

    for epoch in range(1, epoch_limit+1):
        train(mincut_type=mincut_type)
        train_mc, train_o = record_loss(train_loader, mincut_type)

        train_loss = train_mc + train_o

        train_mc_list += [train_mc]
        train_o_list += [train_o]
        train_unsuper_list += [train_loss]

        print("Epoch: "+str(epoch))
        print("training loss: mc "+str(train_mc)+ " o "+str(train_o)+ " total loss "+str(train_loss))

        if epoch == epoch_limit:
            PATH = "_train_model_"+str(epoch)+".pt"
            torch.save(model.state_dict(), model_dir+"/"+PATH)

    df_train = pd.DataFrame(list(zip(train_mc_list,
                                     train_o_list,
                                     train_unsuper_list)))

    df_train.columns = ["train_mc", "train_o", "train_unsupervised"]

    df_train.to_csv(output_dir+"/train_record.csv",
                        index=False)

    # save the embeddings right before softmax out
    model.eval()
    
    layer_outputs = {}

    def hook(module, input, output):
        layer_outputs["linear1"] = output.detach()

    handle = model.linear1.register_forward_hook(hook)

    for data in images_loader:

        layer_outputs.clear()

        data = data.to(device)
        with torch.no_grad():
            _ = model(data.x,
                    remove_self_loops(data.edge_index)[0],
                    data.batch,
                    data.n_cells)

        # Save the desired layer output for linear1 layer
        output = layer_outputs["linear1"]

        df = pd.DataFrame(output.detach().cpu().numpy())

        os.makedirs(embedding_save_dir+"/linear1", exist_ok=True)

        output_filename = "linear1_"+data.region_id[0]+".csv"
        df.to_csv(embedding_save_dir+"/linear1/"+output_filename, index=False)

    handle.remove()

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    mincutpool_run(**vars(args))
