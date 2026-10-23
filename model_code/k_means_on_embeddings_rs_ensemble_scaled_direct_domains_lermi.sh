#!/bin/bash

n_domains=10
n_kmeans_clusters=${n_domains}
epoch_limit=1000
n_repeats=10

output_tag=direct_domains


data_name=lermi_2025
data_dir=../../data/Lermi_data
dist_cutoff=16
group_scheme=default
top_k=4
ctg_comp_dist_cutoff=0.176
degree_limit=20

for graph_type in basic local extended; do
    expanded_edge_cutoff=$((3 * dist_cutoff))
    model_run_id=${data_name}__group_${group_scheme}__base${dist_cutoff}__rad${expanded_edge_cutoff}__topk${top_k}__ctg${ctg_comp_dist_cutoff/./p}__deg${degree_limit}
    log_tag=${graph_type}

    ensemble_run_id=${model_run_id}__ensemble_${n_repeats}

    python k_means_on_embeddings_rs_ensemble_scaled_direct_domains.py \
        --data_name ${data_name} \
        --data_dir ${data_dir} \
        --graph_type ${graph_type} \
        --group_scheme ${group_scheme} \
        --dist_cutoff ${dist_cutoff} \
        --expanded_edge_cutoff ${expanded_edge_cutoff} \
        --top_k ${top_k} \
        --ctg_comp_dist_cutoff ${ctg_comp_dist_cutoff} \
        --degree_limit ${degree_limit} \
        --model_run_id ${model_run_id} \
        --n_kmeans_clusters ${n_kmeans_clusters} \
        --epoch_limit ${epoch_limit} \
        --n_repeats ${n_repeats}

done
