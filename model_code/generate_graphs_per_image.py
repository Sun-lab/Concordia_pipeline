#!/usr/bin/env python
# coding: utf-8

# prepare the graph data objects for each image in the Cords dataset
# generate three graph objects:
# 1. basic graph
# 2. extended graph with two steps extension
# 3. local graph with 1st step extension only

# if using the extended graph
# the second step of graph extension step can take long
# it is recommended to generate the graph object separately for each image
# and move the pyg graph objects together to another folder in later step for model training

# can consider setting random seeds
# though, even if the random seeds are set, the resulting graphs from two runs on the same image
# can be different, due to effect of minor floating number difference on the step of ranking topk cells 
# in 1st step graph extension

import os
import sys
import torch
from torch_geometric.data import Dataset, Data, download_url

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

from scipy.spatial import Delaunay
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

from data_utilities import data_features, make_model_run_id
from graph_data_class import CellularGraphDataset, construct_graph_for_region

import argparse

parser = argparse.ArgumentParser(description='generate extend graph data objects with path purity and downsampling')
parser.add_argument('--data_name', default="cords_2024", type=str, help='the name of the dataset')
parser.add_argument('--data_dir', default="./data/Cords_data", type=str, help='the overall folder of the dataset')
parser.add_argument('--region_index', type=int, default=0, help='the index of the region to generate the object for')
parser.add_argument('--degree_limit', type=int, default=20, help='desired average degree of nodes in each fully extended graph')
parser.add_argument('--dist_cutoff', type=float, default=16, help='distance cutoff for constructing the base graph')
parser.add_argument('--expanded_edge_cutoff', type=float, default=48, help='search radius for candidate cells in the first extension step')
parser.add_argument('--top_k', type=int, default=4, help='number of nearest candidate cells to add in the first extension step')
parser.add_argument('--ctg_comp_dist_cutoff', type=float, default=0.176, help='cell type group composition distance cutoff for adding edges')
parser.add_argument('--group_scheme', type=str, default="default", help='fine-to-coarse cell type grouping scheme')
parser.add_argument('--model_run_id', type=str, default=None, help='dataset-aware identifier for this graph/model parameter set')
parser.add_argument('--prepare_folder', type=str, default="graph_objects_prepare", help='the name of the folder to save the preparation graph files to')                                  

def generate_data(data_name="cords_2024", 
                  data_dir="./data/Cords_data", 
                  region_index=0, 
                  degree_limit=20, 
                  dist_cutoff=16,
                  expanded_edge_cutoff=48,
                  top_k=4,
                  ctg_comp_dist_cutoff=0.176,
                  group_scheme="default",
                  model_run_id=None,
                  prepare_folder="graph_objects_prepare"):

    input_args = locals()
    print("input args are", input_args)

    # Settings
    # load random seeds generated from https://www.random.org
    if os.path.exists(data_dir+"/random_seeds.txt"):
        df_seeds = pd.read_csv(data_dir+"/random_seeds.txt", sep=" ", header=None)
        cur_random_seed = df_seeds[0].tolist()[region_index]
    else:
        cur_random_seed = 0

    random.seed(cur_random_seed)

    if model_run_id is None:
        model_run_id = make_model_run_id(data_name, group_scheme, dist_cutoff,
                                         expanded_edge_cutoff, top_k,
                                         ctg_comp_dist_cutoff, degree_limit)

    data_dicts  = data_features(data_name, "extended",
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

    train_images = data_dicts.train_images
    train_images.sort()

    cur_region_id = train_images[region_index]

    prepare_dir = os.path.join(data_dir, prepare_folder, model_run_id)

    os.makedirs(prepare_dir, exist_ok=True)
    
    raw_data_root = data_dir+"/raw_data"
    dataset_root = prepare_dir+"/"+ cur_region_id

    # Generate cellular graphs from raw inputs
    nx_graph_root = os.path.join(dataset_root, "graph")

    os.makedirs(nx_graph_root, exist_ok=True)


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
        'upto2nd_degree_composition_folder_name': 'group_composition_2nd_basic',
        'processed_folder_name': 'tg_graph_extended',
        'processed_folder_name_basic': 'tg_graph_basic',
        'processed_folder_name_1st': 'tg_graph_local',
        'figure_folder_name': 'figure',
        'node_features': ["cell_type_group", "neighborhood_composition"],
        'neighbor_edge_cutoff': NEIGHBOR_EDGE_CUTOFF,
        'expanded_edge_cutoff': expanded_edge_cutoff,
        'top_k': top_k,
        'degree_limit': degree_limit,
        'ctg_comp_dist_cutoff': ctg_comp_dist_cutoff,
        'path_purity_cutoff': PATH_PURITY_CUTOFF,
        'cell_type_mapping': data_dicts.cell_type_mapping,
        'group_ct_mapping': data_dicts.group_ct_mapping, 
        'operation_type': "build"
    }

    dataset_kwargs.keys()

    dataset_root

    dataset = CellularGraphDataset(dataset_root, **dataset_kwargs)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    generate_data(**vars(args))
