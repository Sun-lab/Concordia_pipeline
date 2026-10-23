#!/bin/bash

mkdir -p merge_clusters_to_domains_Rout

data_name=mpfc_2022
data_dir=../../data/mpfc_data
n_domains=4
n_repeats=10
group_scheme=default
dist_cutoff=380
expanded_edge_cutoff=$((3 * dist_cutoff))
top_k=4
ctg_comp_dist_cutoff=0.176
degree_limit=20

model_run_id=${data_name}__group_${group_scheme}__base${dist_cutoff}__rad${expanded_edge_cutoff}__topk${top_k}__ctg${ctg_comp_dist_cutoff/./p}__deg${degree_limit}

for graph_type in basic local extended; do
for n_kmeans_clusters in 16 10; do

    cur_name=${model_run_id}__rs_ensemble_scaled_${n_repeats}_${graph_type}_clusters_k${n_kmeans_clusters}_domains_k${n_domains}

    R CMD BATCH --quiet --no-save \
    "--args data_name='${data_name}' graph_type='${graph_type}' \
    n_kmeans_clusters='${n_kmeans_clusters}' n_domains='${n_domains}' \
    model_run_id='${model_run_id}' data_dir='${data_dir}' \
    n_repeats='${n_repeats}'" \
    merge_clusters_to_domains_rs_ensemble_scaled_clusters_k.R \
    merge_clusters_to_domains_Rout/merge_clusters_to_domains_${cur_name}.Rout

done
done
