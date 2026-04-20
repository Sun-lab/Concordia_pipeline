#!/bin/bash

for region_index in {0..1533}; do

    python generate_graph_data_object_dist_16_Cords_prepare.py \
          --region_index ${region_index} \
          --degree_limit 20 \
          --graph_type extended \
          --prepare_folder graph_objects_degree_20_prepare
done
