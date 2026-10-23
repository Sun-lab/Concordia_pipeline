#!/bin/bash

for method in extended basic local spatial_lda spatopic; do

    Rscript anova_combined_r2_direct_domains.R "method='${method}'"

done
