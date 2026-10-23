#!/bin/bash

source_data_name=cords_2024
source_data_dir=../../data/Cords_data
target_data_name=lermi_2025
target_data_dir=../../data/Lermi_data
graph_type=extended
group_scheme=default
dist_cutoff=16
expanded_edge_cutoff=$((3 * dist_cutoff))
top_k=4
ctg_comp_dist_cutoff=0.176
degree_limit=20
epoch_limit=1000
n_repeats=10
n_domains=10

source_model_run_id=${source_data_name}__group_${group_scheme}__base${dist_cutoff}__rad${expanded_edge_cutoff}__topk${top_k}__ctg${ctg_comp_dist_cutoff/./p}__deg${degree_limit}
target_model_run_id=${target_data_name}__group_${group_scheme}__base${dist_cutoff}__rad${expanded_edge_cutoff}__topk${top_k}__ctg${ctg_comp_dist_cutoff/./p}__deg${degree_limit}

python transfer_rs_ensemble_scaled_direct_domains.py \
    --source_data_name ${source_data_name} \
    --source_data_dir ${source_data_dir} \
    --target_data_name ${target_data_name} \
    --target_data_dir ${target_data_dir} \
    --graph_type ${graph_type} \
    --group_scheme ${group_scheme} \
    --dist_cutoff ${dist_cutoff} \
    --expanded_edge_cutoff ${expanded_edge_cutoff} \
    --top_k ${top_k} \
    --ctg_comp_dist_cutoff ${ctg_comp_dist_cutoff} \
    --degree_limit ${degree_limit} \
    --source_model_run_id ${source_model_run_id} \
    --target_model_run_id ${target_model_run_id} \
    --gcn_type gat2 \
    --skip_type add \
    --s_dim2 40 \
    --epoch_limit ${epoch_limit} \
    --n_repeats ${n_repeats} \
    --n_domains ${n_domains} \
    --device cpu
