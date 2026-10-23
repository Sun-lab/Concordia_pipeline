#!/usr/bin/env python

import os
import argparse

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from torch_geometric.utils import remove_self_loops

from data_utilities import data_features, make_model_run_id
from models import GCN_model


parser = argparse.ArgumentParser()
parser.add_argument('--source_data_name', default="cords_2024", type=str)
parser.add_argument('--source_data_dir', default="../../data/Cords_data", type=str)
parser.add_argument('--target_data_name', default="lermi_2025", type=str)
parser.add_argument('--target_data_dir', default="../../data/Lermi_data", type=str)
parser.add_argument('--graph_type', default="extended", type=str, choices=['extended', 'basic', 'local'])
parser.add_argument('--group_scheme', default="default", type=str)
parser.add_argument('--dist_cutoff', default=16, type=float)
parser.add_argument('--expanded_edge_cutoff', default=48, type=float)
parser.add_argument('--top_k', default=4, type=int)
parser.add_argument('--ctg_comp_dist_cutoff', default=0.176, type=float)
parser.add_argument('--degree_limit', default=20, type=int)
parser.add_argument('--source_model_run_id', default=None, type=str)
parser.add_argument('--target_model_run_id', default=None, type=str)
parser.add_argument('--gcn_type', default="gat2", type=str, choices=['gcn', 'gat', 'gat2'])
parser.add_argument('--skip_type', default="add", type=str, choices=['no', 'add', 'add2'])
parser.add_argument('--s_dim2', default=40, type=int)
parser.add_argument('--epoch_limit', default=1000, type=int)
parser.add_argument('--n_repeats', default=10, type=int)
parser.add_argument('--n_domains', default=10, type=int)
parser.add_argument('--device', default="gpu", type=str)


def read_embeddings(data_dicts, embedding_dirs):
    cell_ids = []
    blocks = []
    for region in data_dicts.train_images:
        cell_ids.append(pd.read_csv(data_dicts.raw_dir + "/" + region + ".csv",
                                    usecols=["CELL_ID"])["CELL_ID"].to_numpy())
        blocks.append(np.concatenate(
            [pd.read_csv(d + "/linear1/linear1_" + region + ".csv").to_numpy(dtype=np.float32)
             for d in embedding_dirs], axis=1))
    return np.concatenate(cell_ids), np.concatenate(blocks)


def main(args):
    graph_kwargs = dict(group_scheme=args.group_scheme,
                        dist_cutoff=args.dist_cutoff,
                        expanded_edge_cutoff=args.expanded_edge_cutoff,
                        top_k=args.top_k,
                        ctg_comp_dist_cutoff=args.ctg_comp_dist_cutoff,
                        degree_limit=args.degree_limit)
    source_model_run_id = args.source_model_run_id or make_model_run_id(args.source_data_name, **graph_kwargs)
    target_model_run_id = args.target_model_run_id or make_model_run_id(args.target_data_name, **graph_kwargs)

    source_dicts = data_features(args.source_data_name, args.graph_type, model_run_id=source_model_run_id,
                                 data_dir=args.source_data_dir, **graph_kwargs)
    target_dicts = data_features(args.target_data_name, args.graph_type, model_run_id=target_model_run_id,
                                 data_dir=args.target_data_dir, **graph_kwargs)
    assert sorted(source_dicts.group_ct_mapping) == sorted(target_dicts.group_ct_mapping)

    run_ids = [source_model_run_id + "__repeat_" + str(i) for i in range(1, args.n_repeats + 1)]
    ensemble_run_id = source_model_run_id + "__ensemble_" + str(args.n_repeats)
    source_domain_dir = "./results_rs_ensemble_scaled_direct_domains/" + ensemble_run_id + "/" + args.graph_type
    transfer_dir = "./results_rs_transfer/" + target_model_run_id
    epoch_dir = args.graph_type + "/epoch_" + str(args.epoch_limit)
    source_embedding_dirs = ["./results_rs/" + run_id + "/" + epoch_dir for run_id in run_ids]
    target_embedding_dirs = [transfer_dir + "/" + run_id + "/" + epoch_dir for run_id in run_ids]

    device = torch.device("cuda:0" if args.device == "gpu" and torch.cuda.is_available() else "cpu")
    graph_dir = target_dicts.dataset_root + "/" + target_dicts.processed_folder_name

    for run_id, embedding_dir in zip(run_ids, target_embedding_dirs):
        model = GCN_model("comp2nd", len(target_dicts.group_ct_mapping), args.gcn_type, 2,
                          args.s_dim2, "sparse_mincut_pool", args.skip_type)
        model.load_state_dict(torch.load("./saved_models_rs/" + run_id + "/" + args.graph_type +
                                         "/_train_model_" + str(args.epoch_limit) + ".pt",
                                         map_location=device))
        model.to(device).eval()

        outputs = {}
        model.linear1.register_forward_hook(lambda module, input, output: outputs.update(linear1=output))

        os.makedirs(embedding_dir + "/linear1", exist_ok=True)
        for region in target_dicts.train_images:
            data = torch.load(graph_dir + "/" + region + ".gpt").to(device)
            with torch.no_grad():
                model(data.x, remove_self_loops(data.edge_index)[0],
                      torch.zeros(data.num_nodes, dtype=torch.long, device=device), data.num_nodes)
            pd.DataFrame(outputs["linear1"].cpu().numpy()).to_csv(
                embedding_dir + "/linear1/linear1_" + region + ".csv", index=False)

    df_runs = pd.read_csv(source_domain_dir + "/ensemble_runs.csv")
    assert df_runs["model_run_id"].tolist() == run_ids
    scales = np.repeat(df_runs["overall_sd"].to_numpy(dtype=np.float32), args.s_dim2)

    df_source = pd.read_csv(source_domain_dir + "/domains_k" + str(args.n_domains) +
                            "/domain_annotation_for_cells.csv")
    source_ids, source_x = read_embeddings(source_dicts, source_embedding_dirs)
    assert np.array_equal(source_ids, df_source["CELL_ID"].to_numpy())
    source_x /= scales
    source_domains = df_source["domain"].to_numpy()
    centroids = np.stack([source_x[source_domains == d].mean(axis=0, dtype=np.float64)
                          for d in range(1, args.n_domains + 1)])
    del source_x

    target_ids, target_x = read_embeddings(target_dicts, target_embedding_dirs)
    target_x /= scales
    target_domains = cdist(target_x, centroids).argmin(axis=1) + 1

    output_dir = transfer_dir + "/" + ensemble_run_id + "/" + args.graph_type + "/domains_k" + str(args.n_domains)
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame({"CELL_ID": target_ids, "domain": target_domains}).to_csv(
        output_dir + "/domain_annotation_for_cells.csv", index=False)


if __name__ == "__main__":
    main(parser.parse_args())
