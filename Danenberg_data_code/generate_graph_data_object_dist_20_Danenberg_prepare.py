#!/usr/bin/env python
# coding: utf-8

# prepare the graph data objects for the metabric dataset from 
# Danenberg et al. 2022 data
# generate four graph objects in one run:
# 1. basic graph
# 2. extended graph with two steps extension
# 3. extended graph with 1st step extension only
# 4. extended graph with 2nd step extension only

import os
import sys
import torch
from torch_geometric.data import Dataset, Data

import numpy as np
import pandas as pd
import json
import pickle
import matplotlib
import matplotlib.path as mplPath
import matplotlib.pyplot as plt
import networkx as nx
import warnings

from tqdm import tqdm

from scipy.spatial import distance_matrix

from collections.abc import Sequence
from collections import defaultdict
from collections import Counter

from datetime import datetime

import copy

import random

import gc

# this line below is needed if want to import function from other python files in local directory
sys.path.append(os.getcwd())
sys.path.append("../model_code")

from data_utilities import data_features
from graph_data_class import CellularGraphDataset, construct_graph_for_region


import argparse

parser = argparse.ArgumentParser(description='generate extend graph data objects with path purity and downsampling')
parser.add_argument('--region_index', type=int, default=0, help='the index of the region to generate the object for')
parser.add_argument('--degree_limit', type=int, default=20, help='desired average degree of nodes')
parser.add_argument('--graph_type', type=str, default="extended", 
                                    choices=['extended'], help='desired type of cell type group assignment')

def generate_data(region_index=0, degree_limit=20, graph_type='extended'):

    input_args = locals()
    print("input args are", input_args)

    data_dir = "../data/Danenberg_data"
    # Settings
    # load random seeds generated from https://www.random.org
    df_seeds = pd.read_csv(data_dir+"/random_seeds_488.txt", sep=" ", header=None)
    cur_random_seed = df_seeds[0].tolist()[region_index]

    random.seed(cur_random_seed)

    data_dicts  = data_features("danenberg_d20", "both", graph_type)
    train_images = data_dicts.train_images
    train_images.sort()

    cur_region_id = train_images[region_index]

    prepare_dir = data_dir+"/graph_objects_degree_"+str(degree_limit)+"_prepare"

    os.makedirs(prepare_dir, exist_ok=True)
    
    raw_data_root = data_dicts.raw_dir
    dataset_root = prepare_dir+"/"+ cur_region_id

    # Generate cellular graphs from raw inputs
    nx_graph_root = os.path.join(dataset_root, "graph")

    os.makedirs(nx_graph_root, exist_ok=True)

    NEIGHBOR_EDGE_CUTOFF = data_dicts.dist_cutoff
    PATH_PURITY_CUTOFF = data_dicts.path_purity_cutoff
    PATH_LEN_CUTOFF = data_dicts.path_len_cutoff
    TOP_K = 4
    CTG_COMP_DIST_CUTOFF = 0.176

    for region_id in tqdm([cur_region_id]):
        graph_output = os.path.join(nx_graph_root, "%s.gpkl" % region_id)
        if not os.path.exists(graph_output):
            print("Processing %s" % region_id)
            G = construct_graph_for_region(
                region_id,
                cell_data_file=os.path.join(raw_data_root, "%s.csv" % region_id),
                graph_output=graph_output,
                neighbor_edge_cutoff=NEIGHBOR_EDGE_CUTOFF)




    # Define Cellular Graph Dataset
    dataset_kwargs = {
        'raw_cell_info_path': raw_data_root,
        'raw_folder_name': 'graph',
        'niche_folder_name': 'niche_encoded',
        'niche_ct_group_folder_name': 'niche_ct_group_encoded',
        'upto2nd_degree_composition_folder_name': 'group_composition_2nd_basic',
        'processed_folder_name': 'tg_graph',
        'processed_folder_name_basic': 'tg_graph_basic',
        'processed_folder_name_1st': 'tg_graph_1st',
        'processed_folder_name_2nd': 'tg_graph_2nd',
        'figure_folder_name': 'figure',
        'node_features': ["cell_type_group", "neighborhood_composition"],
        'neighbor_edge_cutoff': NEIGHBOR_EDGE_CUTOFF,
        'top_k': TOP_K,
        'degree_limit': degree_limit,
        'ctg_comp_dist_cutoff': CTG_COMP_DIST_CUTOFF,
        'path_purity_cutoff': PATH_PURITY_CUTOFF,
        'path_len_cutoff': PATH_LEN_CUTOFF,
        'cell_type_mapping': data_dicts.cell_type_mapping,
        'group_ct_mapping': data_dicts.group_ct_mapping, 
        'operation_type': 'build'
    }

    dataset_kwargs.keys()

    dataset_root

    dataset = CellularGraphDataset(dataset_root, **dataset_kwargs)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    generate_data(**vars(args))
