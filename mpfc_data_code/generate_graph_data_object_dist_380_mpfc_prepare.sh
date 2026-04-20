#!/bin/bash

for region_index in {0..2}; do
    python  generate_graph_data_object_dist_380_mpfc_prepare.py \
          --region_index ${region_index} \
          --degree_limit 20 \
          --graph_type extended
done
