#!/bin/bash

data_name=mpfc_2022
data_dir=../../data/mpfc_data

results_dir=../../SpatialLDA_copy/results/mpfc_16topics

n_topics=16

for radius in 490 690 1600; do

    for region_index in {0..2}; do
        python cluster_dist_in_image_spatial_lda_celltype_composition.py \
            --data_name ${data_name} \
            --data_dir ${data_dir} \
            --results_dir ${results_dir} \
            --radius ${radius} \
            --n_topics ${n_topics} \
            --region_index ${region_index}
    done

done
