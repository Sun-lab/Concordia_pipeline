#!/bin/bash

first_index=1
last_index=10
random_seed_file=./data/random_seeds_40_triples.txt

lr=0.0001
epoch_limit=1000


data_name=cords_2024
data_dir=../../data/Cords_data
graph_type=extended
group_scheme=default
dist_cutoff=16
top_k=4
ctg_comp_dist_cutoff=0.176
degree_limit=20

for i in $(seq ${first_index} ${last_index}); do
for expanded_edge_cutoff in 32 64; do
    model_run_id=${data_name}__group_${group_scheme}__base${dist_cutoff}__rad${expanded_edge_cutoff}__topk${top_k}__ctg${ctg_comp_dist_cutoff/./p}__deg${degree_limit}
    log_tag=rad${expanded_edge_cutoff}

    run_tag=${model_run_id}__repeat_${i}

    python train_rs.py \
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
        --random_seed_index ${i} \
        --random_seed_file ${random_seed_file} \
        --device gpu \
        --s_dim2 40 \
        --batch_size 64 \
        --lr ${lr} \
        --epoch_limit ${epoch_limit}

done
done


data_name=cords_2024
data_dir=../../data/Cords_data
graph_type=extended
group_scheme=default
dist_cutoff=16
expanded_edge_cutoff=$((3 * dist_cutoff))
ctg_comp_dist_cutoff=0.176
degree_limit=20

for i in $(seq ${first_index} ${last_index}); do
for top_k in 3 5; do
    model_run_id=${data_name}__group_${group_scheme}__base${dist_cutoff}__rad${expanded_edge_cutoff}__topk${top_k}__ctg${ctg_comp_dist_cutoff/./p}__deg${degree_limit}
    log_tag=topk${top_k}

    run_tag=${model_run_id}__repeat_${i}

    python train_rs.py \
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
        --random_seed_index ${i} \
        --random_seed_file ${random_seed_file} \
        --device gpu \
        --s_dim2 40 \
        --batch_size 64 \
        --lr ${lr} \
        --epoch_limit ${epoch_limit}

done
done


data_name=cords_2024
data_dir=../../data/Cords_data
graph_type=extended
group_scheme=default
dist_cutoff=16
expanded_edge_cutoff=$((3 * dist_cutoff))
top_k=4
degree_limit=20

for i in $(seq ${first_index} ${last_index}); do
for ctg_comp_dist_cutoff in 0.202 0.157; do
    model_run_id=${data_name}__group_${group_scheme}__base${dist_cutoff}__rad${expanded_edge_cutoff}__topk${top_k}__ctg${ctg_comp_dist_cutoff/./p}__deg${degree_limit}
    log_tag=ctg${ctg_comp_dist_cutoff/./p}

    run_tag=${model_run_id}__repeat_${i}

    python train_rs.py \
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
        --random_seed_index ${i} \
        --random_seed_file ${random_seed_file} \
        --device gpu \
        --s_dim2 40 \
        --batch_size 64 \
        --lr ${lr} \
        --epoch_limit ${epoch_limit}

done
done


data_name=cords_2024
data_dir=../../data/Cords_data
graph_type=extended
group_scheme=default
dist_cutoff=16
expanded_edge_cutoff=$((3 * dist_cutoff))
top_k=4
ctg_comp_dist_cutoff=0.176

for i in $(seq ${first_index} ${last_index}); do
for degree_limit in 16 24; do
    model_run_id=${data_name}__group_${group_scheme}__base${dist_cutoff}__rad${expanded_edge_cutoff}__topk${top_k}__ctg${ctg_comp_dist_cutoff/./p}__deg${degree_limit}
    log_tag=deg${degree_limit}

    run_tag=${model_run_id}__repeat_${i}

    python train_rs.py \
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
        --random_seed_index ${i} \
        --random_seed_file ${random_seed_file} \
        --device gpu \
        --s_dim2 40 \
        --batch_size 64 \
        --lr ${lr} \
        --epoch_limit ${epoch_limit}

done
done


data_name=cords_2024
data_dir=../../data/Cords_data
graph_type=extended
dist_cutoff=16
expanded_edge_cutoff=$((3 * dist_cutoff))
top_k=4
ctg_comp_dist_cutoff=0.176
degree_limit=20

for i in $(seq ${first_index} ${last_index}); do
for group_scheme in group8; do
    model_run_id=${data_name}__group_${group_scheme}__base${dist_cutoff}__rad${expanded_edge_cutoff}__topk${top_k}__ctg${ctg_comp_dist_cutoff/./p}__deg${degree_limit}
    log_tag=${group_scheme}

    run_tag=${model_run_id}__repeat_${i}

    python train_rs.py \
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
        --random_seed_index ${i} \
        --random_seed_file ${random_seed_file} \
        --device gpu \
        --s_dim2 40 \
        --batch_size 64 \
        --lr ${lr} \
        --epoch_limit ${epoch_limit}

done
done


---- the permuted dataset, a single setting rather than a sweep ----

data_name=cords_permutate
data_dir=../../data/Cords_data_permutate
graph_type=extended
group_scheme=default
dist_cutoff=16
expanded_edge_cutoff=$((3 * dist_cutoff))
top_k=4
ctg_comp_dist_cutoff=0.176
degree_limit=20

for i in $(seq ${first_index} ${last_index}); do
    model_run_id=${data_name}__group_${group_scheme}__base${dist_cutoff}__rad${expanded_edge_cutoff}__topk${top_k}__ctg${ctg_comp_dist_cutoff/./p}__deg${degree_limit}
    log_tag=permutate

    run_tag=${model_run_id}__repeat_${i}

    python train_rs.py \
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
        --random_seed_index ${i} \
        --random_seed_file ${random_seed_file} \
        --device gpu \
        --s_dim2 40 \
        --batch_size 64 \
        --lr ${lr} \
        --epoch_limit ${epoch_limit}

done

