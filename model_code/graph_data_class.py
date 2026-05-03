#!/usr/bin/env python
# coding: utf-8

# include four ways of building graphs:
# basic graph
# two steps extension
# first step extension only
# second step extension only

# part of the code was learnt from 
# https://gitlab.com/enable-medicine-public/space-gm

import os
import torch
from torch_geometric.data import Dataset, Data

import numpy as np
import pandas as pd
import pickle
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import warnings

from scipy.spatial import distance_matrix

from collections.abc import Sequence
from collections import Counter
from collections import defaultdict

from matplotlib.collections import LineCollection

from datetime import datetime

import copy

import random


def load_cell_data(cell_data_file):
    """Load cell coordinates from file

    Args:
        cell_data_file (str): path to csv file containing cell data

    Returns:
        pd.DataFrame: dataframe containing cell coordinates, columns ['CELL_ID', 'X', 'Y', 'CELL_TYPE']
    """
    df = pd.read_csv(cell_data_file)
    df.columns = [c.upper() for c in df.columns]
    assert 'X' in df.columns, "Cannot find column for X coordinates"
    assert 'Y' in df.columns, "Cannot find column for Y coordinates"
    if 'CELL_ID' not in df.columns:
        warnings.warn("Cannot find column for cell id, using index as cell id")
        df['CELL_ID'] = df.index
    return df[['CELL_ID', 'X', 'Y', 'CELL_TYPE']]


def build_graph_from_cell_coords(cell_data, neighbor_edge_cutoff):
    """Construct a networkx graph based on cell coordinates

    Args:
        cell_data (pd.DataFrame): dataframe containing cell data,
            columns ['CELL_ID', 'X', 'Y', ...]
        neighbor_edge_cutoff: the dist cutoff to connect two cells with an edge

    Returns:
        G (nx.Graph): full cellular graph of the region
    """

    coord_ar = np.array(cell_data[['CELL_ID', 'X', 'Y']])
    G = nx.Graph()
    node_to_cell_mapping = {}
    #node_to_ct_mapping = {}

    for i, row in enumerate(coord_ar):
        G.add_node(i)
        node_to_cell_mapping[i] = row[0]
        #node_to_ct_mapping[i] = row[3]

    cur_coords = np.array([[x,y] for x,y in zip(cell_data["X"].tolist(),
                                                cell_data["Y"].tolist())])
    cur_dist_mat = distance_matrix(cur_coords, cur_coords, p=2)
    cur_edges = np.transpose(np.nonzero(cur_dist_mat<=neighbor_edge_cutoff)).tolist()
    cur_edges += [[i, i] for i in range(coord_ar.shape[0])]

    for ij in cur_edges:
        G.add_edge(int(ij[0]), int(ij[1]))

    return G, node_to_cell_mapping


def assign_attributes(G, cell_data, node_to_cell_mapping):
    """Assign node and edge attributes to the cellular graph

    Args:
        G (nx.Graph): full cellular graph of the region
        cell_data (pd.DataFrame): dataframe containing cellular data
        node_to_cell_mapping (dict): 1-to-1 mapping between
            node index in `G` and cell id

    Returns:
        nx.Graph: populated cellular graph
    """
    assert set(G.nodes) == set(node_to_cell_mapping.keys())

    cell_to_node_mapping = {v: k for k, v in node_to_cell_mapping.items()}
    node_properties = {}
    for _, cell_row in cell_data.iterrows():
        cell_id = cell_row['CELL_ID']
        assert cell_id in cell_to_node_mapping, "cell_id not in dictionary"
        node_index = cell_to_node_mapping[cell_id]
        p = {"cell_id": cell_id}
        p["center_coord"] = (cell_row['X'], cell_row['Y'])
        assert "CELL_TYPE" in cell_row
        p["cell_type"] = cell_row["CELL_TYPE"]
        node_properties[node_index] = p

    nx.set_node_attributes(G, node_properties)

    return G


def construct_graph_for_region(region_id,
                               cell_data_file=None,
                               graph_output=None,
                               neighbor_edge_cutoff=None):
    """Construct cellular graph for a region

    Args:
        region_id (str): region id
        cell_data_file (str): path to csv file containing cell coordinates
        graph_output (str): path for saving cellular graph as gpickle
        neighbor_edge_cutoff: the dist cutoff to connect two cells with an edge

    Returns:
        G (nx.Graph): full cellular graph of the region
    """
    assert cell_data_file is not None, "cell data must be provided"
    cell_data = load_cell_data(cell_data_file)
    G, node_to_cell_mapping = build_graph_from_cell_coords(cell_data, neighbor_edge_cutoff)

    # Assign attributes to cellular graph
    G = assign_attributes(G, cell_data, node_to_cell_mapping)
    G.region_id = region_id

    # Save graph to file
    if graph_output is not None:
        with open(graph_output, 'wb') as f:
            pickle.dump(G, f)
    return G


def get_feature_names(features, group_index_mapping):
    """ Helper fn for getting a list of feature names from a list of feature items

    Args:
        features (list): list of feature items
        group_index_mapping (dict): mapping of each cell type group to a unique integer

    Returns:
        feat_names(list): list of feature names
    """
    feat_names = []
    for feat in features:
        if feat in ["cell_type_group"]:
            # feature "cell_type", "edge_type" will be a single integer indice
            # feature "distance" will be a single float value
            feat_names.append(feat)
        elif feat == "neighborhood_composition":
            # feature "neighborhood_composition" will contain a composition vector of the immediate neighbors
            # in addition, a neighborhood composition vector constrained on cell types in each group is also generated
            # The concatenation of the two vectors will have twice the length as the number of unique cell types
            feat_names.extend(["ct_group_composition_1st-%s" % ctg
                               for ctg in sorted(group_index_mapping.keys(), key=lambda x: group_index_mapping[x])])
            feat_names.extend(["ct_group_composition_2nd-%s" % ctg
                               for ctg in sorted(group_index_mapping.keys(), key=lambda x: group_index_mapping[x])])
        else:
            raise ValueError("Feature %s not in allowed options")
    return feat_names



def process_niche_composition(G,
                              node_ind,
                              cell_type_mapping,
                              ct_group_mapping,
                              group_ct_mapping,
                              **kwargs):
    """ Calculate two composition vectors,
        for fine grid cell types and the other for cell type group level

    Args:
        G (nx.Graph): full cellular graph of the region
        node_ind (int): target node index
        cell_type_mapping (dict): mapping of unique cell types to integer indices
        ct_group_mapping (dict): mapping of each unique cell type to the corresponding cell type group
        group_ct_mapping (dict): mapping of each unique cell type group to the set of cell types within it
    Returns:
        ct_composition_vec (list): a vector of niche composition based on fine cell types
        ct_group_omposition_vec (list): a vector of niche composition based on cell type groups
    """


    # first, compute the cell type composition vector on the fine cell type level

    niche_cts = [G.nodes[x]['cell_type'] for x in G.neighbors(node_ind) if x!=node_ind]
    niche_cts += [G.nodes[node_ind]['cell_type']]

    group_list = list(group_ct_mapping.keys())
    group_list.sort()

    group_index_mapping = {}

    for i in range(len(group_list)):
        group_index_mapping[group_list[i]] = i

    niche_ct_ids = [cell_type_mapping[x] for x in niche_cts]
    niche_ct_ids.sort()

    niche_ct_counts = [0 for _ in range(len(cell_type_mapping))]

    for ct_id in niche_ct_ids:
        niche_ct_counts[ct_id] += 1

    ct_composition_vec = [x/len(niche_ct_ids) for x in niche_ct_counts]

    niche_ct_group_ids = [group_index_mapping[ct_group_mapping[x]] for x in niche_cts]
    niche_ct_group_ids.sort()

    niche_ct_group_counts = [0 for _ in range(len(group_list))]

    for ct_group_id in niche_ct_group_ids:
        niche_ct_group_counts[ct_group_id] += 1

    ct_group_composition_vec = [x/len(niche_ct_group_ids) for x in niche_ct_group_counts]

    return ct_composition_vec, ct_group_composition_vec


def process_upto2nd_degree_ct_group_composition(G,
                              node_ind,
                              ct_group_mapping,
                              group_index_mapping):
    """ Calculate one composition vector based on cells in up to 2nd degree neighbors,
        for cell type group level
        by treating all cells in the 2nd degree neighborhood equally
        the output vector will have the same length as the number of cell type groups

    Args:
        G (nx.Graph): full cellular graph of the image region
        node_ind (int): target node index
        ct_group_mapping (dict): mapping of each unique cell type to the corresponding cell type group
        group_index_mapping (dict): mapping of each unique cell type group to the corresponding integer index

    Returns:
        ct_group_composition_upto2nd (list): a vector of cell type group composition 
        based on the 2nd degree neighborhood of the target cell
    """

    first_neighbors = [x for x in G.neighbors(node_ind) if x!=node_ind]

    first_neighbors_list = list(set(first_neighbors+[node_ind]))
    first_neighbors_list.sort()

    second_neighbors_prepare = []

    for x in first_neighbors_list:
        for x_nb in G.neighbors(x):
            if x_nb not in first_neighbors_list:
                second_neighbors_prepare += [x_nb]

    second_neighbors = list(set(second_neighbors_prepare))
    
    all_neighbors = list(set(first_neighbors_list+second_neighbors))
    all_neighbors.sort()
    
    neighbor_cts = [G.nodes[x]['cell_type'] for x in all_neighbors]
    neighbor_ct_group_ids = [group_index_mapping[ct_group_mapping[x]] for x in neighbor_cts]

    neighbor_ct_group_counts = [0 for _ in range(len(group_index_mapping))]

    for group_id in neighbor_ct_group_ids:
        neighbor_ct_group_counts[group_id] += 1

    assert len(neighbor_ct_group_ids) > 0
    ct_group_composition_upto2nd = [x/len(neighbor_ct_group_ids) for x in neighbor_ct_group_counts]

    return ct_group_composition_upto2nd


def process_neighbor_composition(G,
                                 node_ind,
                                 ct_group_mapping,
                                 group_index_mapping, 
                                 **kwargs):
    """ Calculate composition vectors consisting of two parts,
        one part for cell types based on 1st degree neighbors
        another part for cell types based on 2nd degree neighbors

    Args:
        G (nx.Graph): full cellular graph of the region
        node_ind (int): target node index
        ct_group_mapping (dict): mapping of cell types to cell type groups
        group_index_mapping (dict): mapping of each cell type group to a unique integer

    Returns:
        comp_vec (list): a concatenated vector of two composition vectors
    """


    # first, compute the cell type group composition vector based on 1st degree neighbors

    first_neighbors = [x for x in G.neighbors(node_ind) if x!=node_ind]
    neighbor_ctgs_1st = [ct_group_mapping[G.nodes[x]['cell_type']] for x in first_neighbors]

    neighbor_ctg_ids_1st = [group_index_mapping[x] for x in neighbor_ctgs_1st]
    neighbor_ctg_ids_1st.sort()

    neighbor_ctg_counts_1st = [0 for _ in range(len(group_index_mapping))]

    for ctg_id in neighbor_ctg_ids_1st:
        neighbor_ctg_counts_1st[ctg_id] += 1

    if len(neighbor_ctg_ids_1st) == 0:
        composition_vec_1st = [0 for _ in range(len(group_index_mapping))]
    else:
        composition_vec_1st = [x/len(neighbor_ctg_ids_1st) for x in neighbor_ctg_counts_1st]

    # next, compute the cell type group composition vector based on 2nd degree neighbors
    first_neighbors_list = list(set(first_neighbors+[node_ind]))
    first_neighbors_list.sort()

    second_neighbors_prepare = []

    for x in first_neighbors_list:
        for x_nb in G.neighbors(x):
            if x_nb not in first_neighbors_list:
                second_neighbors_prepare += [x_nb]

    second_neighbors = list(set(second_neighbors_prepare))
    second_neighbors.sort()

    neighbor_ctgs_2nd = [ct_group_mapping[G.nodes[x]['cell_type']] for x in second_neighbors]

    neighbor_ctg_ids_2nd = [group_index_mapping[x] for x in neighbor_ctgs_2nd]
    neighbor_ctg_ids_2nd.sort()

    neighbor_ctg_counts_2nd = [0 for _ in range(len(group_index_mapping))]

    for ctg_id in neighbor_ctg_ids_2nd:
        neighbor_ctg_counts_2nd[ctg_id] += 1

    if len(neighbor_ctg_ids_2nd) == 0:
        composition_vec_2nd = [0 for _ in range(len(group_index_mapping))]
    else:
        composition_vec_2nd = [x/len(neighbor_ctg_ids_2nd) for x in neighbor_ctg_counts_2nd]

    return composition_vec_1st+composition_vec_2nd


def process_feature(G, feature_item, node_ind=None, **feature_kwargs):
    """ Process a single node/edge feature item

    The following feature items are supported, note that some of them require
    keyword arguments in `feature_kwargs`:

    Node features:
        - feature_item: "cell_type_group"
            (required) "ct_group_mapping"
            (required) "group_index_mapping"
        - feature_item: "neighborhood_composition"
            (required) "ct_group_mapping"
            (required) "group_index_mapping"

    Args:
        G (nx.Graph): full cellular graph of the region
        feature_item (str): feature item
        node_ind (int): target node index (if feature item is node feature)
        feature_kwargs (dict): arguments for processing features

    Returns:
        v (list): feature vector
    """
    # Node features
    if node_ind is not None:
        if feature_item == "cell_type_group":
            # Integer index of the cell type
            assert "ct_group_mapping" in feature_kwargs, "'ct_group_mapping' is required in the kwargs for feature item 'cell_type_group'"
            assert "group_index_mapping" in feature_kwargs, "'group_index_mapping' is required in the kwargs for feature item 'cell_type_group'"
            v = [feature_kwargs["group_index_mapping"][feature_kwargs["ct_group_mapping"][G.nodes[node_ind]["cell_type"]]]]
            return v
        elif feature_item == "neighborhood_composition":
            # Composition vector consisting of the 1st degree neighbors and 2nd degree neighbors separately
            assert "ct_group_mapping" in feature_kwargs, "'ct_group_mapping' is required in the kwargs for feature item 'neighborhood_composition'"
            assert "group_index_mapping" in feature_kwargs, "'group_index_mapping' is required in the kwargs for feature item 'neighborhood_composition'"
            v = process_neighbor_composition(G, node_ind, **feature_kwargs)
            return v
        else:
            raise ValueError("Feature %s not in allowed options")
    else:
        raise ValueError("One of node_ind or edge_ind should be specified")




def plot_graph(G, file_dir, file_name, group_color_mapping, ct_group_mapping):
    """Plot dot-line graph for the cellular graph

    Args:
        G (nx.Graph): full cellular graph of the region
        file_dir: directory to save the plot
        file_name: name of the file to save the plot
        group_color_mapping (dict): mapping of cell type groups to colors
        ct_group_mapping (dict): mapping of each unique cell type to the corresponding cell type group
    """

    plt.clf()
    plt.figure(figsize=(10, 10))
    # Extract basic node attributes

    node_coords = [G.nodes[n]['center_coord'] for n in G.nodes]
    node_coords = np.stack(node_coords, 0)

    node_colors = [group_color_mapping[ct_group_mapping[G.nodes[n]['cell_type']]] for n in G.nodes]

    assert len(node_colors) == node_coords.shape[0]

    segments = []
    for (i, j, _) in G.edges.data():
        segments.append([G.nodes[i]['center_coord'], G.nodes[j]['center_coord']])

    lc = LineCollection(segments, 
                        colors=(0.4, 0.4, 0.4, 1.0), 
                        linewidths=0.3, 
                        linestyles='--', 
                        zorder=1)

    ax = plt.gca()
    ax.add_collection(lc)

    plt.scatter(node_coords[:, 0],
                node_coords[:, 1],
                s=5,
                c=node_colors,
                linewidths=0.3,
                zorder=2)

    plt.xlim(0, node_coords[:, 0].max() * 1.01)
    plt.ylim(0, node_coords[:, 1].max() * 1.01)

    plt.savefig(file_dir+"/"+file_name+".pdf",
                format="pdf", 
                bbox_inches='tight')

    plt.close()

    return


def nx_to_niche_dataframe(G,
                   **feature_kwargs):
    """ extract the composition vectors based on center cell and 1st degree neighbor
        based on the basic graph
        one vector based on fine grid cell types
        one vector based on cell type groups

    Args:
        G (nx.Graph): full cellular graph of the region
        feature_kwargs (dict): arguments for processing features

    Returns:
        cur_niches (pandas data frame): data frame with each row for one niche around one cell
        cur_niches_ct_group (pandas data frame): data frame with each row for one niche around one cell
    """

    # Append node and edge features to the pyg data object
    cur_niches_list = []
    cur_niches_ct_group_list = []

    for node_ind in G.nodes:
        node_row, node_row_ct_group = process_niche_composition(G, node_ind=node_ind, **feature_kwargs)
        cur_niches_list += [node_row]
        cur_niches_ct_group_list += [node_row_ct_group]

    assert len(cur_niches_list) == G.number_of_nodes()
    assert len(cur_niches_ct_group_list) == G.number_of_nodes()

    inv_map = {v: k for k, v in feature_kwargs['cell_type_mapping'].items()}
    n_cts = len(feature_kwargs['cell_type_mapping'])
    colnames = [inv_map[x] for x in range(n_cts)]
    cur_niches = pd.DataFrame(cur_niches_list, columns = colnames)

    group_list = list(feature_kwargs['group_ct_mapping'].keys())
    group_list.sort()
    cur_niches_ct_group = pd.DataFrame(cur_niches_ct_group_list, columns = group_list)

    return cur_niches, cur_niches_ct_group


def nx_to_upto2nd_degree_ct_group_composition(G,
                   **feature_kwargs):
    """ extract the cell type group composition vectors based on up to 2nd degree neighbors
        for each center cell
        combine the cell type group composition vectors of the center cells into numpy array
        also convert to data frame

    Args:
        G (nx.Graph): full cellular graph of the region
        feature_kwargs (dict): arguments for processing features

    Returns:
        np_up2nd_comp (numpy array): numpy array with each row for one up2nd degree composition vector 
                                     around one cell
        df_up2nd_comp (pandas data frame): data frame with each row for one up2nd degree composition vector 
                                           around one cell
    """

    up2nd_comp_list = []

    for node_ind in G.nodes:
        node_up2nd_comp_ct_group = process_upto2nd_degree_ct_group_composition(G,
                                        node_ind,
                                        feature_kwargs["ct_group_mapping"],
                                        feature_kwargs["group_index_mapping"])
        up2nd_comp_list += [node_up2nd_comp_ct_group]

    assert len(up2nd_comp_list) == G.number_of_nodes()

    inv_map = {v: k for k, v in feature_kwargs['group_index_mapping'].items()}
    n_ct_groups = len(feature_kwargs["group_index_mapping"])
    colnames = [inv_map[x] for x in range(n_ct_groups)]
    np_up2nd_comp = np.array(up2nd_comp_list)
    df_up2nd_comp = pd.DataFrame(up2nd_comp_list, columns = colnames)

    return np_up2nd_comp, df_up2nd_comp


def nx_to_tg_graph(G,
                   node_features=["cell_type_group",
                                  "neighborhood_composition"],
                   **feature_kwargs):
    """ Build pyg data objects from nx graphs

    Args:
        G (nx.Graph): full cellular graph of the region
        node_features (list, optional): list of node feature items
        feature_kwargs (dict): arguments for processing features

    Returns:
        cur_data (pyg graph object): corresponding graph object
    """

    # Append node and edge features to the pyg data object
    cur_data = {"x": [], "edge_index": []}
    for node_ind in G.nodes:
        feat_val = []
        for key in node_features:
            feat_val.extend(process_feature(G, key, node_ind=node_ind, **feature_kwargs))
        cur_data["x"].append(feat_val)

    for edge_ind in G.edges:
        cur_data["edge_index"].append(edge_ind)
        cur_data["edge_index"].append(tuple(reversed(edge_ind)))

    for key, item in cur_data.items():
        cur_data[key] = torch.tensor(item)

    cur_data['edge_index'] = cur_data['edge_index'].t().long()
    cur_data = Data.from_dict(cur_data)

    cur_data.num_nodes = G.number_of_nodes()
    cur_data.region_id = G.region_id

    return cur_data


def nx_to_tg_graph_1st(G,
                   cell_data_file=None,                                    
                   np_comp=None,
                   top_k=4,
                   expanded_edge_cutoff=48,  
                   ctg_comp_dist_cutoff=0.176, 
                   node_features=["cell_type",
                                  "neighborhood_composition"],
                   **feature_kwargs):
    """ Build pyg data objects from nx graphs

    Args:
        G (nx.Graph): full cellular graph of the region
        cell_data_file (str): path to csv file containing raw cell data
        np_comp (numpy array): numpy array of cell type group composition vectors
        top_k: one direction number of closest neighbors in cell type group composition
        expanded_edge_cutoff: expanded dist cutoff for neighbors to consider
        ctg_comp_dist_cutoff: cell type composition dist cutoff for adding edges
        node_features (list, optional): list of node feature items
        feature_kwargs (dict): arguments for processing features

    Returns:
        cur_data (pyg graph object): processed pytorch geometric graph object
    """

    # load basic graph
    # load neighborhood composition information

    # extend graph in only the first step
    # make use of that the cells in the graph object and the rows in neighborhood composition have correspondance

    # add edges from/to cells in neighborhood with extended radius, 
    #   satisfy composition distance requirement, 
    #   among the cells not being existing neighbors in basic graph, 
    #   the topk ones with smallest composition distance to the center cell

    raw_num_nodes = G.number_of_nodes()
    raw_num_edges = G.number_of_edges()

    basic_edges = []
    for edge_ind in G.edges:
        basic_edges.append(edge_ind)

    assert raw_num_edges==len(basic_edges)

    # keep a fixed version to keep track of existing edges in basic graph
    G_fixed_first = copy.deepcopy(G)

    cell_data = load_cell_data(cell_data_file)

    cur_coords = np.array([[x,y] for x,y in zip(cell_data["X"].tolist(),
                                                cell_data["Y"].tolist())])
    cur_dist_mat = distance_matrix(cur_coords, cur_coords, p=2)
    candidate_edges = np.transpose(np.nonzero(cur_dist_mat<=expanded_edge_cutoff)).tolist()

    candidate_neighbors_dict = defaultdict(list)

    for x in candidate_edges:
        candidate_neighbors_dict[x[0]] += [x[1]]

    comp_dist_mat = distance_matrix(np_comp, np_comp, p=2)

    n_cells = cell_data.shape[0]
    assert n_cells == raw_num_nodes
    assert n_cells == G.number_of_nodes()

    for node_id in range(n_cells):

        candidate_neighbors = [x for x in candidate_neighbors_dict[node_id] if x!=node_id]
        
        full_dist_list = comp_dist_mat[node_id].tolist()
        dist_list = [full_dist_list[x] for x in candidate_neighbors]

        sorted_indices = np.argsort(np.array(dist_list)).tolist()

        ordered_neighbors = [candidate_neighbors[x] for x in sorted_indices if dist_list[x]<ctg_comp_dist_cutoff]

        # note here should be G_fixed_first
        node_id_existing_neighbors = set([x for x in G_fixed_first.neighbors(node_id)]) 
        ordered_neighbors_kept = [x for x in ordered_neighbors if x not in node_id_existing_neighbors]    

        if len(ordered_neighbors_kept)>0:
            for j in ordered_neighbors_kept[:top_k]:
                G.add_edge(node_id, j)
                G.add_edge(j, node_id)    

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start adding node and edge features to the pyg data object: ", current_time)

    # Append node and edge features to the pyg data object
    cur_data = {"x": [], "edge_index": []}
    for node_ind in G.nodes:
        feat_val = []
        for key in node_features:
            feat_val.extend(process_feature(G, key, node_ind=node_ind, **feature_kwargs))
        cur_data["x"].append(feat_val)

    for edge_ind in G.edges:
        cur_data["edge_index"].append(edge_ind)
        cur_data["edge_index"].append(tuple(reversed(edge_ind)))

    for key, item in cur_data.items():
        cur_data[key] = torch.tensor(item)

    cur_data['edge_index'] = cur_data['edge_index'].t().long()
    cur_data = Data.from_dict(cur_data)

    cur_data.num_nodes = G.number_of_nodes()
    cur_data.region_id = G.region_id

    return cur_data


def nx_to_tg_graph_2nd(G,
                   cell_data_file=None,                                    
                   np_comp=None,
                   ctg_comp_dist_cutoff=0.176, 
                   degree_limit=None,
                   node_features=["cell_type",
                                  "neighborhood_composition"],
                   figure_dir=None,
                   path_purity_cutoff=None,
                   path_len_cutoff=None,
                   **feature_kwargs):
    """ Build pyg data objects from nx graphs

    Args:
        G (nx.Graph): full cellular graph of the region
        cell_data_file (str): path to csv file containing raw cell data
        np_comp (numpy array): numpy array of cell type group composition vectors
        ctg_comp_dist_cutoff: cell type composition dist cutoff for adding edges
        degree_limit: desired average degree of node
        node_features (list, optional): list of node feature items
        figure_dir: folders to save the figures for the graphs
        path_purity_cutoff: parameter for the quality of the path
        path_len_cutoff: limit for the length of paths to consider
        feature_kwargs (dict): arguments for processing features

    Returns:
        cur_data (pyg graph object): processed pytorch geometric graph object
        raw_num_nodes: the number of codes in the graph
        raw_num_edges: the number of edges in the basic graph
        G.number_of_edges(): the number of edges in the final extended graph
        Counter(added_edge_lens): a counter for the lengths of edges added in the second step only setting
    """

    image_figure_dir = os.path.join(figure_dir, G.region_id)
    os.makedirs(image_figure_dir, exist_ok=True)

    ct_groups = list(feature_kwargs['group_ct_mapping'].keys())
    ct_groups.sort()

    # load basic graph
    # load neighborhood composition information

    # extend graph in the second step only
    # make use of that the cells in the graph object and the rows in neighborhood composition have correspondance

    # add edges based on shortest path quality
    #   two ends have composition distance below cutoff between them
    #   most points on the path do not appear more than the cutoff distance away from either of the two ends

        # compute the number of edges to add in order to reach desired average degree
        # where the degree number is after excluding self-loops, undirectional
        # suppose this number is n_edges_to_add
        # get the total number of additional edges that are qualified to be added, undirectional
            # they need to satisfy source and target cell group requirement
            # they need to satisfy path purity requirement
            # they cannot be existing edges in basic graph
        # suppose this number is n_qualified,
        # build a list of all unique qualified edges and corresponding path lengths
        # compute quartiles of the path length
        # take 10%, 20%, 30%, 40% of n_edges_to_add from each bin
        # right now, if there are not enough edges to sample from,
        # do not take further action to sample from neighboring bins

    raw_num_nodes = G.number_of_nodes()
    raw_num_edges = G.number_of_edges()

    basic_edges = []
    for edge_ind in G.edges:
        basic_edges.append(edge_ind)

    assert raw_num_edges==len(basic_edges)

    # keep a fixed version to keep track of existing edges in basic graph
    G_no_edge = copy.deepcopy(G)
    G_no_edge.clear_edges()

    # this is for only keeping the new edges from the second step only
    # for the purpose of generating plots

    G_add_second = copy.deepcopy(G_no_edge)

    cell_data = load_cell_data(cell_data_file)

    comp_dist_mat = distance_matrix(np_comp, np_comp, p=2)

    n_cells = cell_data.shape[0]
    assert n_cells == raw_num_nodes

    # number of undirectonal edges excluding self-loops
    target_num_edges = G.number_of_nodes() * (degree_limit/2)
    needed_num_edges = target_num_edges - (raw_num_edges-raw_num_nodes)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start computing shortest paths: ", current_time)

    p = dict(nx.shortest_path(G))

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start going through shortest paths: ", current_time)

    if needed_num_edges <= 0:
        print("there are already more than enough edges. do not added edges")
        added_edge_lens = []
    else:
        # undirectional
        candidate_edge_list = []
        candidate_edge_len_list = []

        for i in G.nodes():

            i_targets = list(p[i].keys())

            for j in i_targets:
                if j < i:
                    path_ij = p[i][j]
                    if len(path_ij) <= path_len_cutoff:
                        dist_ij = comp_dist_mat[i,j]
                        if dist_ij < ctg_comp_dist_cutoff:
                            dist_slice = comp_dist_mat[[i,j]][:, path_ij]
                            col_max = np.max(dist_slice, axis=0)
                            cur_purity = np.mean(col_max < ctg_comp_dist_cutoff)
                            if cur_purity > path_purity_cutoff:
                                # add the length of the path to the record if the edge does not exist in the graph
                                # this way will count each edge only in one direction,
                                # since once one direction is added, the other direction will have has_edge being true
                                # given that the edges are non-directional
                                if not G.has_edge(i, j):
                                    candidate_edge_list += [(i, j)]
                                    candidate_edge_len_list += [len(path_ij)-1]

        # bin the candidate edges into four bins

        np_candidate_edge_len = np.array(candidate_edge_len_list)

        percentile_list = []
        percentile_list += [0]
        percentile_list += [np.percentile(np_candidate_edge_len, 25, interpolation='linear')]
        percentile_list += [np.percentile(np_candidate_edge_len, 50, interpolation='linear')]
        percentile_list += [np.percentile(np_candidate_edge_len, 75, interpolation='linear')]
        percentile_list += [max(candidate_edge_len_list)]

        bin_dict = {}
        for i in range(4):
            bin_dict[i] = []

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Start binning paths: ", current_time)

        for (i,j), ij_len in list(zip(candidate_edge_list, candidate_edge_len_list)):
            if ij_len <= percentile_list[1]:
                bin_dict[0] += [((i,j), ij_len)]
            elif ij_len <= percentile_list[2]:
                bin_dict[1] += [((i,j), ij_len)]
            elif ij_len <= percentile_list[3]:
                bin_dict[2] += [((i,j), ij_len)]
            else:
                bin_dict[3] += [((i,j), ij_len)]

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Start adding edges: ", current_time)

        # compute the number of edges needed to add
        needed_num_edges_bins = [round(0.1*needed_num_edges),
                                 round(0.2*needed_num_edges),
                                 round(0.3*needed_num_edges)]
        needed_num_edges_bins += [int(needed_num_edges - sum(needed_num_edges_bins))]


        added_edge_lens = []

        for bin_i in range(len(bin_dict)):
            bin_i_list = bin_dict[bin_i]
            random.shuffle(bin_i_list)
            bin_i_taken = bin_i_list[:needed_num_edges_bins[bin_i]]
            for (i,j), ij_len in bin_i_taken:
                G.add_edge(i, j)
                G_add_second.add_edge(i, j)
                added_edge_lens += [ij_len]

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Start generating figures for extended graphs: ", current_time)

        # generate figure for additional edges

        file_name=G_add_second.region_id+"_2nd_only_additional_edges"
        plot_graph(G_add_second,
                   image_figure_dir,
                   file_name,
                   feature_kwargs['group_color_mapping'],
                   feature_kwargs['ct_group_mapping'])
        
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Start generating figures for extended graphs: ", current_time)

        # generate figure for extended graph

        file_name=G.region_id+"_2nd_only_extended_graph"
        plot_graph(G,
                   image_figure_dir,
                   file_name,
                   feature_kwargs['group_color_mapping'],
                   feature_kwargs['ct_group_mapping'])

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start adding node and edge features to the pyg data object: ", current_time)

    # Append node and edge features to the pyg data object
    cur_data = {"x": [], "edge_index": []}
    for node_ind in G.nodes:
        feat_val = []
        for key in node_features:
            feat_val.extend(process_feature(G, key, node_ind=node_ind, **feature_kwargs))
        cur_data["x"].append(feat_val)

    for edge_ind in G.edges:
        cur_data["edge_index"].append(edge_ind)
        cur_data["edge_index"].append(tuple(reversed(edge_ind)))

    for key, item in cur_data.items():
        cur_data[key] = torch.tensor(item)

    cur_data['edge_index'] = cur_data['edge_index'].t().long()
    cur_data = Data.from_dict(cur_data)

    cur_data.num_nodes = G.number_of_nodes()
    cur_data.region_id = G.region_id

    return cur_data, raw_num_nodes, raw_num_edges, G.number_of_edges(), Counter(added_edge_lens)




def nx_to_tg_graph_shortest_path_expand_degree_limit(G,
                   cell_data_file=None,                                    
                   np_comp=None,
                   expanded_edge_cutoff=48, 
                   top_k=4, 
                   ctg_comp_dist_cutoff=0.176, 
                   degree_limit=None,
                   node_features=["cell_type_group",
                                  "neighborhood_composition"],
                   figure_dir=None,
                   path_purity_cutoff=None,
                   path_len_cutoff=None,
                   **feature_kwargs):
    """ Build pyg data objects from nx graphs

    Args:
        G (nx.Graph): full cellular graph of the region
        cell_data_file (str): path to csv file containing raw cell data
        np_comp (numpy array): numpy array of cell type group composition vectors
        expanded_edge_cutoff: expanded dist cutoff for neighbors to consider
        top_k: one direction number of closest neighbors in cell type group composition
        ctg_comp_dist_cutoff: cell type composition dist cutoff for adding edges
        degree_limit: desired average degree of node
        node_features (list, optional): list of node feature items
        figure_dir: folders to save the figures for the graphs
        path_purity_cutoff: parameter for the quality of the path
        path_len_cutoff: limit for the length of paths to consider
        feature_kwargs (dict): arguments for processing features

    Returns:
        cur_data (pyg graph object): processed pytorch geometric graph object
        raw_num_nodes: the number of nodes in the basic graph
        raw_num_edges: the number of edges in the basic graph
        first_num_edges: the number of edges in the first extended graph
        G.number_of_edges(): the number of edges in the final extended graph
        Counter(added_edge_lens): a counter for the lengths of edges added in the second step
    """

    image_figure_dir = os.path.join(figure_dir, G.region_id)
    os.makedirs(image_figure_dir, exist_ok=True)

    ct_groups = list(feature_kwargs['group_ct_mapping'].keys())
    ct_groups.sort()

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start plotting basic graph: ", current_time)

    file_name=G.region_id+"_basic_graph"
    plot_graph(G,
               image_figure_dir,
               file_name,
               feature_kwargs['group_color_mapping'],
               feature_kwargs['ct_group_mapping'])


    # load basic graph
    # load neighborhood composition information

    # extend graph in two steps
    # make use of that the cells in the graph object and the rows in neighborhood composition have correspondance

    # add edges from/to cells in neighborhood with extended radius, 
    #   satisfy composition distance requirement, 
    #   among the cells not being existing neighbors in basic graph, 
    #   the topk ones with smallest composition distance to the center cell
    # add edges based on shortest path quality
    #   two ends have composition distance below cutoff between them
    #   most points on the path do not appear more than the cutoff distance away from either of the two ends

        # compute the number of edges to add in order to reach desired average degree
        # where the degree number is after excluding self-loops, undirectional
        # suppose this number is n_edges_to_add
        # get the total number of additional edges that are qualified to be added, undirectional
            # they need to satisfy source and target cell group requirement
            # they need to satisfy path purity requirement
            # they cannot be existing edges in basic graph
        # suppose this number is n_qualified,
        # build a list of all unique qualified edges and corresponding path lengths
        # compute quartiles of the path length
        # take 10%, 20%, 30%, 40% of n_edges_to_add from each bin
        # right now, if there are not enough edges to sample from,
        # do not take further action to sample from neighboring bins

    raw_num_nodes = G.number_of_nodes()
    raw_num_edges = G.number_of_edges()

    basic_edges = []
    for edge_ind in G.edges:
        basic_edges.append(edge_ind)

    assert raw_num_edges==len(basic_edges)

    # keep a fixed version to keep track of existing edges in basic graph
    G_fixed_first = copy.deepcopy(G)

    G_no_edge = copy.deepcopy(G)
    G_no_edge.clear_edges()

    # these are for only keeping the new edges from the two steps
    # for the purpose of generating plots

    G_add_first = copy.deepcopy(G_no_edge)
    G_add_second = copy.deepcopy(G_no_edge)

    cell_data = load_cell_data(cell_data_file)

    cur_coords = np.array([[x,y] for x,y in zip(cell_data["X"].tolist(),
                                                cell_data["Y"].tolist())])
    cur_dist_mat = distance_matrix(cur_coords, cur_coords, p=2)
    candidate_edges = np.transpose(np.nonzero(cur_dist_mat<=expanded_edge_cutoff)).tolist()

    candidate_neighbors_dict = defaultdict(list)

    for x in candidate_edges:
        candidate_neighbors_dict[x[0]] += [x[1]]

    comp_dist_mat = distance_matrix(np_comp, np_comp, p=2)

    n_cells = cell_data.shape[0]
    assert n_cells == raw_num_nodes
    assert n_cells == G.number_of_nodes()

    for node_id in range(n_cells):

        candidate_neighbors = [x for x in candidate_neighbors_dict[node_id] if x!=node_id]
        
        full_dist_list = comp_dist_mat[node_id].tolist()
        dist_list = [full_dist_list[x] for x in candidate_neighbors]

        sorted_indices = np.argsort(np.array(dist_list)).tolist()

        ordered_neighbors = [candidate_neighbors[x] for x in sorted_indices if dist_list[x]<ctg_comp_dist_cutoff]

        # note here should be G_fixed_first
        node_id_existing_neighbors = set([x for x in G_fixed_first.neighbors(node_id)]) 
        ordered_neighbors_kept = [x for x in ordered_neighbors if x not in node_id_existing_neighbors]    

        if len(ordered_neighbors_kept)>0:
            for j in ordered_neighbors_kept[:top_k]:
                G.add_edge(node_id, j)
                G.add_edge(j, node_id)    
                G_add_first.add_edge(node_id, j)
                G_add_first.add_edge(j, node_id)  

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start plotting first additional edges: ", current_time)

    file_name=G_add_first.region_id+"_first_additional_edges"
    plot_graph(G_add_first,
               image_figure_dir,
               file_name,
               feature_kwargs['group_color_mapping'],
               feature_kwargs['ct_group_mapping'])

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start plotting first extended graph: ", current_time)

    file_name=G.region_id+"_first_extended_graph"
    plot_graph(G,
               image_figure_dir,
               file_name,
               feature_kwargs['group_color_mapping'],
               feature_kwargs['ct_group_mapping'])

    
    # number of undirectonal edges excluding self-loops
    first_num_edges = G.number_of_edges()
    target_num_edges = G.number_of_nodes() * (degree_limit/2)
    needed_num_edges = target_num_edges - (first_num_edges-raw_num_nodes)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start computing shortest paths: ", current_time)

    p = dict(nx.shortest_path(G))

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start going through shortest paths: ", current_time)

    if needed_num_edges <= 0:
        print("there are already more than enough edges. do not added edges")
        added_edge_lens = []
    else:
        # undirectional
        candidate_edge_list = []
        candidate_edge_len_list = []

        for i in G.nodes():

            i_targets = list(p[i].keys())

            for j in i_targets:
                if j < i:
                    path_ij = p[i][j]
                    if len(path_ij) <= path_len_cutoff:
                        dist_ij = comp_dist_mat[i,j]
                        if dist_ij < ctg_comp_dist_cutoff:
                            dist_slice = comp_dist_mat[[i,j]][:, path_ij]
                            col_max = np.max(dist_slice, axis=0)
                            cur_purity = np.mean(col_max < ctg_comp_dist_cutoff)
                            if cur_purity > path_purity_cutoff:
                                # add the length of the path to the record if the edge does not exist in the graph
                                # this way will count each edge only in one direction,
                                # since once one direction is added, the other direction will have has_edge being true
                                # given that the edges are non-directional
                                if not G.has_edge(i, j):
                                    candidate_edge_list += [(i, j)]
                                    candidate_edge_len_list += [len(path_ij)-1]

        # bin the candidate edges into four bins

        np_candidate_edge_len = np.array(candidate_edge_len_list)

        percentile_list = []
        percentile_list += [0]
        percentile_list += [np.percentile(np_candidate_edge_len, 25, interpolation='linear')]
        percentile_list += [np.percentile(np_candidate_edge_len, 50, interpolation='linear')]
        percentile_list += [np.percentile(np_candidate_edge_len, 75, interpolation='linear')]
        percentile_list += [max(candidate_edge_len_list)]

        bin_dict = {}
        for i in range(4):
            bin_dict[i] = []

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Start binning paths: ", current_time)

        for (i,j), ij_len in list(zip(candidate_edge_list, candidate_edge_len_list)):
            if ij_len <= percentile_list[1]:
                bin_dict[0] += [((i,j), ij_len)]
            elif ij_len <= percentile_list[2]:
                bin_dict[1] += [((i,j), ij_len)]
            elif ij_len <= percentile_list[3]:
                bin_dict[2] += [((i,j), ij_len)]
            else:
                bin_dict[3] += [((i,j), ij_len)]

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Start adding edges: ", current_time)

        # compute the number of edges needed to add
        needed_num_edges_bins = [round(0.1*needed_num_edges),
                                 round(0.2*needed_num_edges),
                                 round(0.3*needed_num_edges)]
        needed_num_edges_bins += [int(needed_num_edges - sum(needed_num_edges_bins))]


        added_edge_lens = []

        for bin_i in range(len(bin_dict)):
            bin_i_list = bin_dict[bin_i]
            random.shuffle(bin_i_list)
            bin_i_taken = bin_i_list[:needed_num_edges_bins[bin_i]]
            for (i,j), ij_len in bin_i_taken:
                G.add_edge(i, j)
                G_add_second.add_edge(i, j)
                added_edge_lens += [ij_len]

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Start generating figures for extended graphs: ", current_time)

        # generate figure for additional edges

        file_name=G_add_second.region_id+"_second_additional_edges"
        plot_graph(G_add_second,
                   image_figure_dir,
                   file_name,
                   feature_kwargs['group_color_mapping'],
                   feature_kwargs['ct_group_mapping'])
        
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Start generating figures for extended graphs: ", current_time)

        # generate figure for extended graph

        file_name=G.region_id+"_second_extended_graph"
        plot_graph(G,
                   image_figure_dir,
                   file_name,
                   feature_kwargs['group_color_mapping'],
                   feature_kwargs['ct_group_mapping'])

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start adding node and edge features to the pyg data object: ", current_time)

    # Append node and edge features to the pyg data object
    cur_data = {"x": [], "edge_index": []}
    for node_ind in G.nodes:
        feat_val = []
        for key in node_features:
            feat_val.extend(process_feature(G, key, node_ind=node_ind, **feature_kwargs))
        cur_data["x"].append(feat_val)

    for edge_ind in G.edges:
        cur_data["edge_index"].append(edge_ind)
        cur_data["edge_index"].append(tuple(reversed(edge_ind)))

    for key, item in cur_data.items():
        cur_data[key] = torch.tensor(item)

    cur_data['edge_index'] = cur_data['edge_index'].t().long()
    cur_data = Data.from_dict(cur_data)

    cur_data.num_nodes = G.number_of_nodes()
    cur_data.region_id = G.region_id

    return cur_data, raw_num_nodes, raw_num_edges, first_num_edges, G.number_of_edges(), Counter(added_edge_lens)





class CellularGraphDataset(Dataset):
    """ Main dataset structure for cellular graphs
    Inherited from https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Dataset.html
    """

    def __init__(self,
                 root,
                 transform=[],
                 pre_transform=None,
                 raw_cell_info_path='raw_data',
                 raw_folder_name='graph',
                 niche_folder_name='niche_encoded',
                 niche_ct_group_folder_name='niche_ct_group_encoded',
                 upto2nd_degree_composition_folder_name='group_composition_2nd_basic',
                 processed_folder_name='tg_graph',
                 processed_folder_name_basic='tg_graph_basic',
                 processed_folder_name_1st='tg_graph_1st',   
                 processed_folder_name_2nd='tg_graph_2nd',                 
                 figure_folder_name='figure',
                 node_features=["cell_type_group", "neighborhood_composition"],
                 neighbor_edge_cutoff=16,
                 top_k=4,
                 degree_limit = 20,
                 ctg_comp_dist_cutoff=0.176,
                 path_purity_cutoff=0.9,
                 path_len_cutoff = 30000,
                 cell_type_mapping=None,
                 group_ct_mapping=None, 
                 operation_type="load"):
        """ Initialize the dataset

        Args:
            root (str): path to the dataset directory
            transform (list): list of transformations (see `transform.py`),
                applied to each output graph on-the-fly
            pre_transform (list): list of transformations, applied to each graph before saving
            raw_cell_info_path (str): path to the raw cell info csv file
            raw_folder_name (str): name of the sub-folder containing raw nx graphs
            niche_folder_name (str): name of the sub-folder containing cell type composition based on 1st degree neighborhood in basic graph
            niche_ct_group_folder_name (str): name of the sub-folder containing cell type group composition based on 1st degree neighborhood in basic graph
            upto2nd_degree_composition_folder_name (str): name of the sub-folder containing cell type group composition based on up to 2nd degree neighborhood in basic graph
            processed_folder_name (str): name of the sub-folder containing processed graphs (pyg data object) based on the graph after two steps of extensions
            processed_folder_name_basic (str): name of the sub-folder containing processed graphs (pyg data object) based on the basic graph
            processed_folder_name_1st (str): name of the sub-folder containing processed graphs (pyg data object) based on the graph only after the first step of extension
            processed_folder_name_2nd (str): name of the sub-folder containing processed graphs (pyg data object) based on the graph only after the second step of extension
            figure_folder_name (str): name of the sub-folder containing figures for graphs
            node_features (list): list of feature items to get for the nodes
            neighbor_edge_cutoff (int): distance cutoff used to decide neighbors in the basic graph
            top_k (int): for the 1st graph extension, number of closest neighbors to consider in terms of distance in cell type group composition
            degree_limit (int): desired average degree of the graph after two steps of extensions
            ctg_comp_dist_cutoff (float): cell type group composition distance cutoff for adding edges
            path_purity_cutoff (float): purity cutoff for the paths to consider
            path_len_cutoff (int): length cutoff for the paths to consider
            cell_type_mapping (dict): mapping of each unique cell type to a unique integer
            group_ct_mapping (dict): mapping of each cell type group to a set of cell types in that group
            operation_type (str): "load" or "build", whether to load the processed graphs if they exist, or to build the processed graphs by calling `process` function
        """
        self.root = root
        self.raw_cell_info_path = raw_cell_info_path
        self.raw_folder_name = raw_folder_name
        self.niche_folder_name = niche_folder_name
        self.niche_ct_group_folder_name = niche_ct_group_folder_name
        self.upto2nd_degree_composition_folder_name = upto2nd_degree_composition_folder_name
        self.processed_folder_name = processed_folder_name
        self.processed_folder_name_basic = processed_folder_name_basic
        self.processed_folder_name_1st = processed_folder_name_1st
        self.processed_folder_name_2nd = processed_folder_name_2nd
        self.figure_folder_name = figure_folder_name
        self.graph_metric_folder_name = "graph_metrics"
        self.operation_type = operation_type
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.niche_dir, exist_ok=True)
        os.makedirs(self.niche_ct_group_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.processed_basic_dir, exist_ok=True)
        os.makedirs(self.processed_1st_dir, exist_ok=True)
        os.makedirs(self.processed_2nd_dir, exist_ok=True)
        os.makedirs(self.figure_dir, exist_ok=True)
        os.makedirs(self.metric_dir, exist_ok=True)
        os.makedirs(self.upto2nd_degree_composition_dir, exist_ok=True)

        self.cell_type_mapping = cell_type_mapping
        self.group_ct_mapping = group_ct_mapping
        self.neighbor_edge_cutoff = neighbor_edge_cutoff
        self.top_k = top_k
        self.degree_limit = degree_limit
        self.ctg_comp_dist_cutoff = ctg_comp_dist_cutoff
        self.path_purity_cutoff = path_purity_cutoff
        self.path_len_cutoff = path_len_cutoff

        group_keys = list(group_ct_mapping.keys())
        group_keys.sort()

        if len(group_keys)<=9:
            group_color_mapping = {group_name: matplotlib.colormaps.get_cmap("Set1")(i)
                                   for i, group_name in enumerate(group_keys)}
        elif len(group_keys)<=12:
            group_color_mapping = {group_name: matplotlib.colormaps.get_cmap("Paired")(i)
                                   for i, group_name in enumerate(group_keys)}
        else:
            group_color_mapping = {group_name: matplotlib.colormaps.get_cmap("tab20")(i%20)
                                   for i, group_name in enumerate(group_keys)}
                                   
        self.group_color_mapping = group_color_mapping

        ct_group_mapping = {}

        for cur_group_key in group_keys:
            cur_set = group_ct_mapping[cur_group_key]
            cur_list = list(cur_set)
            cur_list.sort()
            for cur_ct in cur_list:
                ct_group_mapping[cur_ct] = cur_group_key

        assert set(ct_group_mapping.keys()) == set(cell_type_mapping.keys())
        self.ct_group_mapping = ct_group_mapping

        group_index_mapping = {}

        for i in range(len(group_keys)):
            group_index_mapping[group_keys[i]] = i
        
        self.group_index_mapping = group_index_mapping

        # Node features
        self.node_features = node_features

        if "cell_type_group" in self.node_features:
            assert self.node_features.index("cell_type_group") == 0, "cell_type_group must be the first node feature"

        self.node_feature_names = get_feature_names(node_features,
                                                    group_index_mapping=self.group_index_mapping)

        # Prepare kwargs for hold dictionaries for dealing with features
        self.feature_kwargs = {}
        self.feature_kwargs['cell_type_mapping'] = self.cell_type_mapping
        self.feature_kwargs['group_ct_mapping'] = self.group_ct_mapping
        self.feature_kwargs['ct_group_mapping'] = self.ct_group_mapping
        self.feature_kwargs['group_color_mapping'] = self.group_color_mapping
        self.feature_kwargs['group_index_mapping'] = self.group_index_mapping

        # Note this command below calls the `process` function
        super(CellularGraphDataset, self).__init__(root, None, pre_transform)

        # Transformations, e.g. masking features, adding graph-level labels
        self.transform = transform

        # Cache for graphs
        self.cached_data = {}

        self.N = len(self.processed_paths)
        # 
        self.region_ids = [self.get_full(idx).region_id for idx in range(self.N)]


    def set_transforms(self, transform=[]):
        """Set transformation functions"""
        self.transform = transform

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.raw_folder_name)

    @property
    def niche_dir(self) -> str:
        return os.path.join(self.root, self.niche_folder_name)

    @property
    def niche_ct_group_dir(self) -> str:
        return os.path.join(self.root, self.niche_ct_group_folder_name)

    @property
    def upto2nd_degree_composition_dir(self) -> str:
        return os.path.join(self.root, self.upto2nd_degree_composition_folder_name)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.processed_folder_name)

    @property
    def processed_basic_dir(self) -> str:
        return os.path.join(self.root, self.processed_folder_name_basic)

    @property
    def processed_1st_dir(self) -> str:
        return os.path.join(self.root, self.processed_folder_name_1st)

    @property
    def processed_2nd_dir(self) -> str:
        return os.path.join(self.root, self.processed_folder_name_2nd)

    @property
    def figure_dir(self) -> str:
        return os.path.join(self.root, self.figure_folder_name)

    @property
    def metric_dir(self) -> str:
        return os.path.join(self.root, self.graph_metric_folder_name)

    @property
    def raw_file_names(self):
        return sorted([f for f in os.listdir(self.raw_dir) if f.endswith('.gpkl')])

    @property
    def processed_file_names(self):
        # Only files for full graphs
        return sorted([f for f in os.listdir(self.processed_dir) if f.endswith('.gpt')])

    def len(self):
        return self.N

    def process(self):
        """Featurize all cellular graphs"""
        if self.operation_type=="build":

            for raw_path in self.raw_paths:
                G = pickle.load(open(raw_path, 'rb'))
                region_id = G.region_id

                print("niche composition files exist or not?")
                niche_flag = os.path.exists(os.path.join(self.niche_dir, '%s.csv' % region_id))
                niche_ct_group_flag = os.path.exists(os.path.join(self.niche_ct_group_dir, '%s.csv' % region_id))
                print(niche_flag)
                print(niche_ct_group_flag)

                if (not niche_flag) or (not niche_ct_group_flag):
                    # save out the 1st neigbor niche encoding vector files out for each image
                    niche_dataframe, niche_ct_group_dataframe = nx_to_niche_dataframe(G,
                                                                **self.feature_kwargs)

                    assert niche_dataframe.shape[1] == len(self.cell_type_mapping)  # make sure feature dimension matches
                    assert niche_ct_group_dataframe.shape[1] == len(self.group_ct_mapping)  # make sure feature dimension matches

                    niche_dataframe.to_csv(os.path.join(self.niche_dir, '%s.csv' % region_id),
                                        index=False)

                    niche_ct_group_dataframe.to_csv(os.path.join(self.niche_ct_group_dir, '%s.csv' % region_id),
                                                    index=False)

                print("up to 2nd degree composition files exist or not?")
                print(os.path.exists(os.path.join(self.upto2nd_degree_composition_dir, '%s.csv' % region_id)))
                if not os.path.exists(os.path.join(self.upto2nd_degree_composition_dir, '%s.csv' % region_id)):
                    # get the numpy array of up to 2nd degree neighbor composition to use for building extended graphs
                    # also save out the up to 2nd degree neighbor composition files out for each image
                    np_upto2nd, df_upto2nd = nx_to_upto2nd_degree_ct_group_composition(G,
                                                                                    **self.feature_kwargs)

                    assert df_upto2nd.shape[1] == len(self.group_ct_mapping)  # make sure feature dimension matches

                    df_upto2nd.to_csv(os.path.join(self.upto2nd_degree_composition_dir, '%s.csv' % region_id),
                                                                index=False)
                else:
                    df_upto2nd = pd.read_csv(os.path.join(self.upto2nd_degree_composition_dir, '%s.csv' % region_id),
                                            header=0)
                    np_upto2nd = df_upto2nd.to_numpy()

                print("basic graph exist or not?")
                print(os.path.exists(os.path.join(self.processed_basic_dir, '%s.gpt' % region_id)))
                if not os.path.exists(os.path.join(self.processed_basic_dir, '%s.gpt' % region_id)):
                    # Transform networkx graphs to pyg data objects, and add features for nodes and edges
                    graph_data = nx_to_tg_graph(G,
                                                node_features=self.node_features,
                                                **self.feature_kwargs)

                    assert graph_data.region_id == region_id  # graph identifier
                    print(graph_data.x.shape)
                    print(len(self.node_feature_names))
                    assert graph_data.x.shape[1] == len(self.node_feature_names)  # make sure feature dimension matches

                    torch.save(graph_data, os.path.join(self.processed_basic_dir, '%s.gpt' % graph_data.region_id))

                print("extended graph exist or not?")
                print(os.path.exists(os.path.join(self.processed_dir, '%s.gpt' % region_id)))
                if not os.path.exists(os.path.join(self.processed_dir, '%s.gpt' % region_id)):
                    # Transform networkx graphs to pyg data objects, and add features for nodes and edges
                    graph_data, raw_num_nodes, raw_num_edges, first_num_edges, final_num_edges, counter_added_lens = \
                        nx_to_tg_graph_shortest_path_expand_degree_limit(G,
                                                                    cell_data_file=os.path.join(self.raw_cell_info_path, '%s.csv' % region_id),
                                                                    np_comp=np_upto2nd, 
                                                                    top_k=self.top_k,
                                                                    expanded_edge_cutoff=3*self.neighbor_edge_cutoff,
                                                                    ctg_comp_dist_cutoff=self.ctg_comp_dist_cutoff,
                                                                    degree_limit=self.degree_limit,
                                                                    node_features=self.node_features,
                                                                    figure_dir=self.figure_dir,
                                                                    path_purity_cutoff=self.path_purity_cutoff,
                                                                    path_len_cutoff=self.path_len_cutoff,
                                                                    **self.feature_kwargs)
                    
                    assert graph_data.region_id == region_id  # graph identifier
                    print(graph_data.x.shape)
                    print(len(self.node_feature_names))
                    assert graph_data.x.shape[1] == len(self.node_feature_names)  # make sure feature dimension matches

                    torch.save(graph_data, os.path.join(self.processed_dir, '%s.gpt' % graph_data.region_id))

                    region_metric_dir = os.path.join(self.metric_dir, region_id)
                    os.makedirs(region_metric_dir, exist_ok=True)

                    df_raw_nums = pd.DataFrame(list(zip([raw_num_nodes], [raw_num_edges])),
                                                    columns=["raw_num_nodes", "raw_num_edges"])
                    df_raw_nums.to_csv(os.path.join(region_metric_dir, "raw_num_nodes_edges.csv"),
                                            index=False)

                    df_extended_nums = pd.DataFrame(list(zip([raw_num_nodes], [raw_num_edges], 
                                                            [first_num_edges], [final_num_edges])),
                                                    columns=["raw_num_nodes", "raw_num_edges", 
                                                            "first_num_edges", "final_num_edges"])
                    df_extended_nums.to_csv(os.path.join(region_metric_dir, "extended_num_edges.csv"),
                                            index=False)
                    
                    df_len_counter = pd.DataFrame.from_dict(counter_added_lens, orient='index').reset_index()
                    df_len_counter = df_len_counter.rename(columns={'index':'len', 0:'count'})
                    df_len_counter.to_csv(os.path.join(region_metric_dir, "path_length_counter.csv"),
                                            index=False)

                print("1st extended only graph exist or not?")
                print(os.path.exists(os.path.join(self.processed_1st_dir, '%s.gpt' % region_id)))
                if not os.path.exists(os.path.join(self.processed_1st_dir, '%s.gpt' % region_id)):
                    # delete and reload the graph to remove the edges added in the second step
                    del G
                    G = pickle.load(open(raw_path, 'rb'))
                    region_id = G.region_id
                    # generate graph object based on 1st extension
                    # transform networkx graphs to pyg data objects and add features for nodes and edges
                    graph_1st_only = nx_to_tg_graph_1st(G,
                                                    cell_data_file=os.path.join(self.raw_cell_info_path, '%s.csv' % region_id),
                                                    np_comp=np_upto2nd, 
                                                    top_k=self.top_k,
                                                    expanded_edge_cutoff=3*self.neighbor_edge_cutoff,
                                                    ctg_comp_dist_cutoff=self.ctg_comp_dist_cutoff,
                                                    node_features=self.node_features,
                                                    **self.feature_kwargs)
                    assert graph_1st_only.region_id == region_id  # graph identifier
                    print(graph_1st_only.x.shape)
                    print(len(self.node_feature_names))
                    assert graph_1st_only.x.shape[1] == len(self.node_feature_names)
                    torch.save(graph_1st_only, os.path.join(self.processed_1st_dir, '%s.gpt' % graph_1st_only.region_id))

                print("2nd extended only graph exist or not?")
                print(os.path.exists(os.path.join(self.processed_2nd_dir, '%s.gpt' % region_id)))
                if not os.path.exists(os.path.join(self.processed_2nd_dir, '%s.gpt' % region_id)):
                    # delete and reload the graph to remove the edges added in the first step
                    del G
                    G = pickle.load(open(raw_path, 'rb'))
                    region_id = G.region_id
                    # generate graph object based on 2nd extension
                    # transform networkx graphs to pyg data objects and add features for nodes and edges

                    graph_2nd_only, raw_num_nodes, raw_num_edges, final_num_edges, counter_added_lens = nx_to_tg_graph_2nd(G,
                                                        cell_data_file=os.path.join(self.raw_cell_info_path, '%s.csv' % region_id),
                                                        np_comp=np_upto2nd, 
                                                        ctg_comp_dist_cutoff=self.ctg_comp_dist_cutoff,
                                                        degree_limit=self.degree_limit,
                                                        node_features=self.node_features,
                                                        figure_dir=self.figure_dir,
                                                        path_purity_cutoff=self.path_purity_cutoff,
                                                        path_len_cutoff=self.path_len_cutoff,
                                                        **self.feature_kwargs)
                    assert graph_2nd_only.region_id == region_id  # graph identifier
                    print(graph_2nd_only.x.shape)
                    print(len(self.node_feature_names))
                    assert graph_2nd_only.x.shape[1] == len(self.node_feature_names)
                    torch.save(graph_2nd_only, os.path.join(self.processed_2nd_dir, '%s.gpt' % graph_2nd_only.region_id))

                    region_metric_dir = os.path.join(self.metric_dir, region_id)
                    os.makedirs(region_metric_dir, exist_ok=True)

                    df_raw_nums = pd.DataFrame(list(zip([raw_num_nodes], [raw_num_edges])),
                                                    columns=["raw_num_nodes", "raw_num_edges"])
                    df_raw_nums.to_csv(os.path.join(region_metric_dir, "2nd_only_raw_num_nodes_edges.csv"),
                                            index=False)

                    df_extended_nums = pd.DataFrame(list(zip([raw_num_nodes], [raw_num_edges], [final_num_edges])),
                                                    columns=["raw_num_nodes", "raw_num_edges", "final_num_edges"])
                    df_extended_nums.to_csv(os.path.join(region_metric_dir, "2nd_only_extended_num_edges.csv"),
                                            index=False)
                    
                    df_len_counter = pd.DataFrame.from_dict(counter_added_lens, orient='index').reset_index()
                    df_len_counter = df_len_counter.rename(columns={'index':'len', 0:'count'})
                    df_len_counter.to_csv(os.path.join(region_metric_dir, "2nd_only_path_length_counter.csv"),
                                            index=False)

        return

    def __getitem__(self, i): # i is the relative index within a dataset or a subset
        """Sample a graph from the dataset and apply transformations"""
        if (isinstance(i, (int, np.integer))
            or (isinstance(i, Tensor) and i.dim() == 0)
            or (isinstance(i, np.ndarray) and np.isscalar(i))):

            data = self.get(self.indices()[i])
            # Apply transformations
            for transform_fn in self.transform:
                data = transform_fn(data)
            return data
        else:
            return self.index_select(i)

    def get(self, idx):
        data = self.get_full(idx)
        return data

    def get_full(self, idx):
        """Read the full cellular graph of region `idx`"""
        # note that here we assign the image label again at this step
        # to allow the flexibility in assigning labels
        if idx in self.cached_data:
            return self.cached_data[idx]
        else:
            data = torch.load(self.processed_paths[idx])
            self.cached_data[idx] = data
            return data

    def get_full_nx(self, idx):
        """Read the full cellular graph (nx.Graph) of region `idx`"""
        return pickle.load(open(self.raw_paths[idx], 'rb'))

    def clear_cache(self):
        del self.cached_data
        self.cached_data = {}
        return

    def index_select(self, i_list: Sequence) -> 'Dataset':
        r"""Creates a subset of the dataset from specified indices :obj:`idx`.
        Indices :obj:`idx` can be a slicing object, *e.g.*, :obj:`[2:5]`, a
        list, a tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type
        long or bool."""
        indices = self.indices()

        if isinstance(i_list, Sequence) and not isinstance(i_list, str):
            indices = [indices[i] for i in i_list]

        else:
            raise IndexError(
                f"Only list is valid indice format "
                f"(got "
                f"'{type(idx).__name__}')")

        dataset = copy.copy(self)
        subset_region_ids = []
        for _, idx in enumerate(indices):
            data = self.get(idx)
            subset_region_ids += [data.region_id]
        dataset.region_ids = subset_region_ids
        dataset.N = len(subset_region_ids)
        dataset._indices = indices

        return dataset
