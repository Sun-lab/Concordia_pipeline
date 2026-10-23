#!/bin/bash

mkdir -p merge_clusters_to_domains_spatopic_Rout

data_name=mpfc_2022
data_dir=../../data/mpfc_data

results_dir=../../SpaTopic_copy/results/mpfc_16topics

n_kmeans_clusters=16

n_domains=4

for setting in "490 22" "690 26" "1600 40"; do

    set -- ${setting}
    radius=$1
    sigma=$2

    cur_name=spatopic_${n_kmeans_clusters}topics_radius_${radius}_sigma_${sigma}_domains_k${n_domains}

    echo "merging radius ${radius} sigma ${sigma} into ${n_domains} domains"

    R CMD BATCH --quiet --no-save \
    "--args data_name='${data_name}' data_dir='${data_dir}' \
results_dir='${results_dir}' radius='${radius}' sigma='${sigma}' \
n_kmeans_clusters='${n_kmeans_clusters}' n_domains='${n_domains}'" \
    merge_clusters_to_domains_spatopic.R \
    merge_clusters_to_domains_spatopic_Rout/merge_clusters_to_domains_${cur_name}.Rout

done
