#!/bin/bash


data_name=mpfc_2022
data_dir=../../data/mpfc_data

results_dir=../../SpaTopic_copy/results/mpfc_16topics

n_topics=16

for setting in "490 22" "690 26" "1600 40"; do

    set -- ${setting}
    radius=$1
    sigma=$2

    for region_index in {0..2}; do
        python cluster_dist_in_image_spatopic_celltype_composition.py \
            --data_name ${data_name} \
            --data_dir ${data_dir} \
            --results_dir ${results_dir} \
            --radius ${radius} \
            --sigma ${sigma} \
            --n_topics ${n_topics} \
            --region_index ${region_index}
    done

done
