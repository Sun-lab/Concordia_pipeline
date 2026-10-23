#!/bin/bash

mkdir -p merge_clusters_to_domains_spatial_lda_Rout

data_name=mpfc_2022
data_dir=../../data/mpfc_data

results_dir=../../SpatialLDA_copy/results/mpfc_16topics

n_kmeans_clusters=16

n_domains=4

for radius in 490 690 1600; do

    cur_name=spatial_lda_${n_kmeans_clusters}topics_radius_${radius}_domains_k${n_domains}

    echo "merging radius ${radius} into ${n_domains} domains"

    R CMD BATCH --quiet --no-save \
    "--args data_name='${data_name}' data_dir='${data_dir}' \
    results_dir='${results_dir}' radius='${radius}' \
    n_kmeans_clusters='${n_kmeans_clusters}' n_domains='${n_domains}'" \
    merge_clusters_to_domains_spatial_lda.R \
    merge_clusters_to_domains_spatial_lda_Rout/merge_clusters_to_domains_${cur_name}.Rout

done
