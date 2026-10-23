# find a proper distance cutoff for Tim's data Panel 1

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

import random

from tqdm import tqdm

from scipy.spatial import Delaunay
from scipy.spatial import distance_matrix

from collections.abc import Sequence
from collections import defaultdict
from collections import Counter

from datetime import datetime


# this line below is needed if want to import function from other python files in local directory
sys.path.append(os.getcwd())
sys.path.append("../model_code")

from data_utilities import data_features
from graph_data_class import load_cell_data, build_graph_from_cell_coords



data_dicts  = data_features("mpfc_d20", "all", "extended")
train_images = data_dicts.train_images
train_images.sort()

random.seed(1152)

random_images = train_images

raw_data_root = data_dicts.raw_dir

neighbor_cutoff_list = [360, 380, 390, 400, 460, 480, 490, 500, 
                        560 ,600, 660, 670, 680, 690, 700, 720, 
                        900, 1200, 1500, 1600]

for nb_cutoff in neighbor_cutoff_list:

    ave_degrees = []

    for image in random_images:
        fname = os.path.join(raw_data_root, image+".csv")
        cell_data = load_cell_data(fname)
        G, _ = build_graph_from_cell_coords(cell_data, nb_cutoff)
        G_degree = (G.number_of_edges() - G.number_of_nodes())/G.number_of_nodes()
        ave_degrees += [2*G_degree]

    print("nb_cutoff "+str(nb_cutoff)+": "+str(np.median(np.array(ave_degrees))))

# 3 images 
# nb_cutoff 360: 5.313632030505243
# nb_cutoff 380: 5.958055290753098
# nb_cutoff 390: 6.305052430886558
# nb_cutoff 400: 6.627264061010486

# picked cutoff: 
# 380