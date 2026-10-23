#!/bin/bash

n_domains=10
n_kmeans_clusters=${n_domains}
epoch_limit=1000
n_repeats=10

output_tag=direct_domains


data_name=cords_2024
data_dir=../../data/Cords_data
graph_type=extended
group_scheme=default
dist_cutoff=16
top_k=4
ctg_comp_dist_cutoff=0.176
degree_limit=20

for expanded_edge_cutoff in 32 64; do
    model_run_id=${data_name}__group_${group_scheme}__base${dist_cutoff}__rad${expanded_edge_cutoff}__topk${top_k}__ctg${ctg_comp_dist_cutoff/./p}__deg${degree_limit}
    log_tag=rad${expanded_edge_cutoff}

    ensemble_run_id=${model_run_id}__ensemble_${n_repeats}

    python k_means_on_embeddings_rs_ensemble_scaled_direct_domains_sensitivity.py \
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


data_name=cords_2024
data_dir=../../data/Cords_data
graph_type=extended
group_scheme=default
dist_cutoff=16
expanded_edge_cutoff=$((3 * dist_cutoff))
ctg_comp_dist_cutoff=0.176
degree_limit=20

for top_k in 3 5; do
    model_run_id=${data_name}__group_${group_scheme}__base${dist_cutoff}__rad${expanded_edge_cutoff}__topk${top_k}__ctg${ctg_comp_dist_cutoff/./p}__deg${degree_limit}
    log_tag=topk${top_k}

    ensemble_run_id=${model_run_id}__ensemble_${n_repeats}

    python k_means_on_embeddings_rs_ensemble_scaled_direct_domains_sensitivity.py \
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


data_name=cords_2024
data_dir=../../data/Cords_data
graph_type=extended
group_scheme=default
dist_cutoff=16
expanded_edge_cutoff=$((3 * dist_cutoff))
top_k=4
degree_limit=20

for ctg_comp_dist_cutoff in 0.202 0.157; do
    model_run_id=${data_name}__group_${group_scheme}__base${dist_cutoff}__rad${expanded_edge_cutoff}__topk${top_k}__ctg${ctg_comp_dist_cutoff/./p}__deg${degree_limit}
    log_tag=ctg${ctg_comp_dist_cutoff/./p}

    ensemble_run_id=${model_run_id}__ensemble_${n_repeats}

    python k_means_on_embeddings_rs_ensemble_scaled_direct_domains_sensitivity.py \
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


data_name=cords_2024
data_dir=../../data/Cords_data
graph_type=extended
group_scheme=default
dist_cutoff=16
expanded_edge_cutoff=$((3 * dist_cutoff))
top_k=4
ctg_comp_dist_cutoff=0.176

for degree_limit in 16 24; do
    model_run_id=${data_name}__group_${group_scheme}__base${dist_cutoff}__rad${expanded_edge_cutoff}__topk${top_k}__ctg${ctg_comp_dist_cutoff/./p}__deg${degree_limit}
    log_tag=deg${degree_limit}

    ensemble_run_id=${model_run_id}__ensemble_${n_repeats}

    python k_means_on_embeddings_rs_ensemble_scaled_direct_domains_sensitivity.py \
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


data_name=cords_2024
data_dir=../../data/Cords_data
graph_type=extended
dist_cutoff=16
expanded_edge_cutoff=$((3 * dist_cutoff))
top_k=4
ctg_comp_dist_cutoff=0.176
degree_limit=20

for group_scheme in group8; do
    model_run_id=${data_name}__group_${group_scheme}__base${dist_cutoff}__rad${expanded_edge_cutoff}__topk${top_k}__ctg${ctg_comp_dist_cutoff/./p}__deg${degree_limit}
    log_tag=${group_scheme}

    ensemble_run_id=${model_run_id}__ensemble_${n_repeats}

    python k_means_on_embeddings_rs_ensemble_scaled_direct_domains_sensitivity.py \
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


data_name=cords_permutate
data_dir=../../data/Cords_data_permutate
graph_type=extended
group_scheme=default
dist_cutoff=16
expanded_edge_cutoff=$((3 * dist_cutoff))
top_k=4
ctg_comp_dist_cutoff=0.176
degree_limit=20

model_run_id=${data_name}__group_${group_scheme}__base${dist_cutoff}__rad${expanded_edge_cutoff}__topk${top_k}__ctg${ctg_comp_dist_cutoff/./p}__deg${degree_limit}
log_tag=permutate

ensemble_run_id=${model_run_id}__ensemble_${n_repeats}

python k_means_on_embeddings_rs_ensemble_scaled_direct_domains_sensitivity.py \
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
