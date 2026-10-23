#!/bin/bash

for method in extended basic local spatial_lda spatopic; do

    python morans_i_in_image_direct_domains.py --method ${method}

done
