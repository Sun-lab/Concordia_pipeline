#!/bin/bash

SUBTYPE=both
N_CLUSTERS=40


for data_name in danenberg_d20; do
  degree_limit=20
  for cell_feature in comp2nd; do
    for mincut_type in sparse_mincutpool; do
      for subtype in ${SUBTYPE}; do
        for graph_type in extended basic 1st; do
          for predictor_type in prop; do
            for loss_type in unsupervised; do
              for gcn_type in gat2; do
                for skip_type in add; do
                  for device in gpu; do
                    for lr in 0.0001; do
                      for n_clusters in ${N_CLUSTERS}; do
                        for n_gcns in 2; do
                          for o1_weight in 1.0; do
                            for o2_weight in 1; do
                              for batch_size in 64; do
                                for epoch_limit in 1000; do
                                  python train.py \
                                        --data_name ${data_name} \
                                        --subtype ${subtype} \
                                        --graph_type ${graph_type} \
                                        --cell_feature ${cell_feature} \
                                        --mincut_type ${mincut_type} \
                                        --predictor_type ${predictor_type} \
                                        --loss_type ${loss_type} \
                                        --gcn_type ${gcn_type} \
                                        --skip_type ${skip_type} \
                                        --device ${device} \
                                        --n_clusters ${n_clusters} \
                                        --n_gcns ${n_gcns} \
                                        --o1_weight ${o1_weight} \
                                        --o2_weight ${o2_weight} \
                                        --batch_size ${batch_size} \
                                        --lr ${lr} \
                                        --epoch_limit ${epoch_limit} \
                                        --degree_limit ${degree_limit}
                                done
                              done
                            done
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
