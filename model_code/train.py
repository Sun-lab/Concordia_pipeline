#!/usr/bin/env python
# coding: utf-8

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

#from tqdm.notebook import tqdm
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

from data_utilities import data_features
from models import GCN_model

from data_transformers import compute_ct_proportion, add_num_of_cells
from sparse_unsupervised_pooling import sparse_mincutpool
from graph_data_class import CellularGraphDataset


parser = argparse.ArgumentParser(description='train GNN model with unsupervised loss')

parser.add_argument('--data_name', default="cords_d20", type=str, help='the name of the dataset to use')
parser.add_argument('--subtype', default="LUAD", type=str, help='a string to indicate which subset to use',
                                 choices=['LUAD', 'LUSC', 'both', 'ERpos', 'ERneg', 'panel1', 'all'])
parser.add_argument('--graph_type', default="extended", type=str,
                                    help='name for the feature and graph combination.',
                                    choices=['extended', 'basic', '1st'])
parser.add_argument('--cell_feature', default="comp2nd", type=str, help='the kind of cell feature to use',
                                      choices=['ct', 'comp', 'comp2nd'])
parser.add_argument('--mincut_type', default='sparse_mincutpool', type=str,
                                     help='the tpye of mincutpool layer to use',
                                     choices=['sparse_mincutpool'])
parser.add_argument('--predictor_type', default="prop", type=str,
                                     help='the type of predictor to connect to final prediction',
                                     choices=['prop', 'atilde', 'ave' ,'flat', 'atildeflat', 'centroidatildeflat'])
parser.add_argument('--loss_type', default="unsupervised", type=str,
                                   choices=['unsupervised'],
                                   help='use unsupervised loss only')
parser.add_argument('--gcn_type', default="gcn", type=str,
                                   choices=['gcn', 'gat', 'gat2'],
                                   help='what gcn layer to use')
parser.add_argument('--skip_type', default="no", type=str,
                                   choices=['no', 'add', 'concat', 'add2', 'concat2'],
                                   help='which type of skip connection to use')
parser.add_argument('--device', default="gpu", type=str, help='whether to use CPU or GPU')
parser.add_argument('--n_clusters', default=10, type=int, help='number of clusters')
parser.add_argument('--n_gcns', default=3, type=int, help='number of GCNs in integer format')
parser.add_argument('--o1_weight', default=1.0, type=float, help='weight of o1 loss in float format')
parser.add_argument('--o2_weight', default=1.0, type=float, help='weight of o2 loss in float format')
parser.add_argument('--batch_size', default=2, type=int, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, for example, 0.001')
parser.add_argument('--epoch_limit', type=int, default=1000, help='max number of epoches, for example, 1000')
parser.add_argument('--degree_limit', type=int, default=20, help='the average degree to achieve for each image, for example, 20')


def mincutpool_run(data_name="cords_d20", subtype="LUAD", graph_type="extended", cell_feature="comp", 
                   mincut_type="sparse_three_entropy", predictor_type="prop",
                   loss_type="unsupervised", gcn_type="gcn", skip_type="no", device="gpu", n_clusters=10, n_gcns=3,
                   o1_weight=1.0, o2_weight=1.0, batch_size=2,
                   lr=0.001, epoch_limit=200, degree_limit=20):

    torch.manual_seed(1629)
    random.seed(1000)
    np.random.seed(1028)

    input_args = locals()
    print("input args are", input_args)

    # Metadata for the chosen dataset
    # not split on dataset based on patients into training/validation/test
    # all images go to training set
    data_dicts  = data_features(data_name, subtype, graph_type)
    
    NEIGHBOR_EDGE_CUTOFF = data_dicts.dist_cutoff
    PATH_PURITY_CUTOFF = data_dicts.path_purity_cutoff
    PATH_LEN_CUTOFF = data_dicts.path_len_cutoff


    unsupervised_pooling_ready = ['sparse_mincutpool']
    assert mincut_type in unsupervised_pooling_ready, "input mincut_type must be one of "+" ".join(unsupervised_pooling_ready)

    fast_dir = "../"

    data_subfolder = data_dicts.data_subfolder

    output_dir = fast_dir+"results/"+data_subfolder+"/"+graph_type+"/"+data_name+"_"+cell_feature+"_"+subtype+"_"+ \
                 graph_type+"_"+mincut_type+"_"+predictor_type+"_"+loss_type+"_"+gcn_type+"/"+ \
                 data_name+"_"+cell_feature+"_"+subtype+"_"+mincut_type+"_"+predictor_type+"_"+ \
                 loss_type+"_"+gcn_type+"_"+skip_type+"_"+device+"_clusters_"+str(n_clusters)+"_gcn_"+str(n_gcns)+"_1_o1_"+str(o1_weight)+ \
                 "_o2_"+str(o2_weight)+"_batch_"+str(batch_size)+ \
                 "_lr_"+str(lr)[2:]+"_epoch_"+str(epoch_limit)

    model_dir = fast_dir+"saved_models/"+data_subfolder+"/"+graph_type+"/"+data_name+"_"+cell_feature+"_"+subtype+"_"+ \
                graph_type+"_"+mincut_type+"_"+predictor_type+"_"+loss_type+"_"+gcn_type+"/"+ \
                data_name+"_"+cell_feature+"_"+subtype+"_"+mincut_type+"_"+predictor_type+"_"+ \
                loss_type+"_"+gcn_type+"_"+skip_type+"_"+device+"_clusters_"+str(n_clusters)+"_gcn_"+str(n_gcns)+"_1_o1_"+str(o1_weight)+ \
                "_o2_"+str(o2_weight)+"_batch_"+str(batch_size)+ \
                "_lr_"+str(lr)[2:]+"_epoch_"+str(epoch_limit)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Settings
    raw_data_root = data_dicts.raw_dir
    dataset_root = data_dicts.dataset_root

    if device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)


    patients = data_dicts.patients
    train_images = data_dicts.train_images

    print("The number of patients involved: ")
    print(len(patients))
    print("The number of images involved: ")
    print(len(train_images))


    # Define Cellular Graph Dataset
    dataset_kwargs = {
        'raw_cell_info_path': raw_data_root,
        'raw_folder_name': 'graph',
        'processed_folder_name': data_dicts.processed_folder_name,
        'node_features': ["cell_type_group", "neighborhood_composition"],
        'neighbor_edge_cutoff': NEIGHBOR_EDGE_CUTOFF,
        'degree_limit': degree_limit,
        'path_purity_cutoff': PATH_PURITY_CUTOFF,
        'path_len_cutoff': PATH_LEN_CUTOFF,
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

    # transformers = []
    # dataset.set_transforms(transformers)

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

    train_s_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)


    model = GCN_model(cell_feature, N_CELL_TYPE_GROUPS,
                      gcn_type, n_gcns, n_clusters, mincut_type, predictor_type, skip_type)
    print(model)
    model.to(device)

    #model = GCN(hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    def train(pre_flag, loss_type, mincut_type):
        model.zero_grad(set_to_none=True)
        # model.to(device)
        model.train()

        cnt = 0
        for data in train_loader:  # Iterate in batches over the training dataset.
            
            data = data.to(device)
            if mincut_type in ["sparse_mincutpool"]:
                out, s_batched, mc_loss, o1_loss = model(data.x,
                                                         remove_self_loops(data.edge_index)[0],
                                                         data.batch,
                                                         data.n_cells)   # Perform a single forward pass.

            if pre_flag:
                if mincut_type in ["sparse_mincutpool"]:
                    loss = mc_loss + (o1_weight*o1_loss)  # Compute the loss.

            else:
                sys.exit("Input argument loss_type is not among the defined ones.")

            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            model.zero_grad(set_to_none=True)  # Clear gradients.


    def record_loss(loader, mincut_type):

        model.to(device)
        model.eval()

        mc_loss_list = []
        o1_loss_list = []
        o2_loss_list = []

        with torch.no_grad():
            for data in loader:  # Iterate in batches over the training/validation/test dataset.

                data = data.to(device)

                if mincut_type in ["sparse_mincutpool"]:

                    out, s_batched, mc_loss, o1_loss = model(data.x,
                                                             remove_self_loops(data.edge_index)[0],
                                                             data.batch,
                                                             data.n_cells)

                    mc_loss_value = mc_loss.to('cpu').item()
                    o1_loss_value = o1_loss.to('cpu').item()

                    mc_loss_list += [mc_loss_value]
                    o1_loss_list += [o1_loss_value]
                    o2_loss_list += [0]


        mc_ave = sum(mc_loss_list)/len(mc_loss_list)
        o1_ave = sum(o1_loss_list)/len(o1_loss_list)
        o2_ave = sum(o2_loss_list)/len(o2_loss_list)
        #print(cur_auc)
        return mc_ave, o1_ave, o2_ave


    def save_scores(loader, model_name, mincut_type):

        model.to(device)
        model.eval()

        os.makedirs(output_dir+"/"+model_name+"_cluster_scores", exist_ok=True)

        with torch.no_grad():

            for data in loader:  # Iterate image by image over the training/validation/test dataset.

                data = data.to(device)

                if mincut_type in ["sparse_mincutpool"]:
                    out, s_batched, mc_loss, o1_loss = model(data.x,
                                                             remove_self_loops(data.edge_index)[0],
                                                             data.batch,
                                                             data.n_cells)

                df = pd.DataFrame(s_batched.squeeze().detach().cpu().numpy())
                df.columns = ["cluster_"+str(i) for i in range(n_clusters)]

                cur_cluster_filename = data.region_id[0]+".csv"

                df.to_csv(output_dir+"/"+model_name+"_cluster_scores/"+cur_cluster_filename, index=False)


    if device != "cpu":
        print("before training: ")
        print(f"gpu used {torch.cuda.max_memory_allocated(device=None)} memory")


    if loss_type in ["unsupervised"]:

        train_mc_list = []
        train_o1_list = []
        train_o2_list = []
        train_unsuper_list = []

        for epoch in range(1, epoch_limit+1):
            train(pre_flag=True, loss_type=loss_type, mincut_type=mincut_type)
            train_mc, train_o1, train_o2 = record_loss(train_loader, mincut_type)

            train_loss = (train_mc + (o1_weight*train_o1) + (o2_weight*train_o2))

            train_mc_list += [train_mc]
            train_o1_list += [train_o1]
            train_o2_list += [train_o2]
            train_unsuper_list += [train_loss]

            print("Epoch: "+str(epoch))
            print("training loss: mc "+str(train_mc)+ " o1 "+str(train_o1)+" o2 "+str(train_o2)+" total loss "+str(train_loss))

            PATH = "_pretrain_model_"+str(epoch)+".pt"
            torch.save(model.state_dict(), model_dir+"/"+PATH)

        df_pretrain = pd.DataFrame(list(zip(train_mc_list,
                                            train_o1_list,
                                            train_o2_list,
                                            train_unsuper_list)))

        df_pretrain.columns = ["train_mc", "train_o1", "train_o2", "train_unsupervised"]

        df_pretrain.to_csv(output_dir+"/pretrain_record.csv",
                          index=False)

        save_scores(train_s_loader, "pretrain", mincut_type)


        if cell_feature not in ["comp", "comp2nd"]:
            df_embedding = pd.DataFrame(model.ct_embedding.weight.detach().cpu().numpy().copy())
            df_embedding.to_csv(output_dir+"/pretrain_ct_embedding.csv", index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    mincutpool_run(**vars(args))
