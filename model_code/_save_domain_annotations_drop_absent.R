save_domain_annotations_drop_absent <- function(dir, domain_dir, region_ids, n_kmeans_clusters, 
                                n_cells_threshold, n_images_threshold, 
                                n_domains){
  dir.create(domain_dir, recursive=TRUE, showWarnings=FALSE)
  
  cluster_file = file.path(dir, "kmeans_cluster.csv")
  df_cluster = read.csv(cluster_file, header=TRUE)
  dim(df_cluster)  
  head(df_cluster)
  
  table(df_cluster$kmeans_cluster)
  
  embed_list = list()
  coord_list = list()
  
  for (cur_region in region_ids){
    
    df_embed_cur = read.csv(file.path(dir, "embedding_dist_in_image", 
                                      sprintf("embedding_dist_%s.csv", cur_region)), 
                            header=FALSE)
    embed_list[[cur_region]] = as.matrix(df_embed_cur)
    
    df_coord_cur = read.csv(file.path(dir, "coord_dist_in_image", 
                                      sprintf("coord_dist_%s.csv", cur_region)), 
                            header=FALSE)  
    coord_list[[cur_region]] = as.matrix(df_coord_cur)
    
  }
  
  count_clusters_hard = read.csv(file.path(dir, paste0("cluster_hard_count_kmeans.csv")), 
                                 header=TRUE)
  
  all_cluster_names = colnames(count_clusters_hard)[2:(n_kmeans_clusters+1)]
  
  n_cells_total = sapply(all_cluster_names, function(c_name){
    sum(count_clusters_hard[[c_name]])
  })
  
  n_images_present = sapply(all_cluster_names, function(c_name){
    sum(count_clusters_hard[[c_name]] >= n_cells_threshold)
  })
  
  keep_names = all_cluster_names[n_cells_total > 0]
  drop_names = all_cluster_names[n_cells_total == 0]
  
  keep_index = which(n_cells_total > 0)
  
  cat("clusters holding at least one cell: ", length(keep_names), " of ",
      n_kmeans_clusters, "\n", sep="")
  
  still_absent = keep_names[n_images_present[keep_names] == 0]
  if (length(still_absent) > 0){
    cat("NOTE: ", length(still_absent), " kept cluster(s) hold cells but never reach ",
        n_cells_threshold, " in any image: ", paste(still_absent, collapse=", "),
        "\n", sep="")
    cat("      they co-occur with nothing, so every one of their pairs is still ",
        "interpolated\n      to 2 * max and each can still take a domain of its own\n",
        sep="")
  }
  
  if (length(drop_names) > 0){
    
    cat("dropping ", length(drop_names), " EMPTY clusters: ",
        paste(drop_names, collapse=", "), "\n", sep="")
    cat("  they hold no cells, so NO CELL LOSES ITS DOMAIN by this filter\n", sep="")
    
    write.csv(data.frame(cluster = drop_names,
                         n_cells = as.integer(n_cells_total[drop_names]),
                         n_images_present = as.integer(n_images_present[drop_names])),
              file.path(domain_dir, "dropped_clusters.csv"),
              row.names=FALSE)
  }else{
    cat("every cluster holds at least one cell; nothing dropped\n")
    write.csv(data.frame(cluster = character(0),
                         n_cells = integer(0),
                         n_images_present = integer(0)),
              file.path(domain_dir, "dropped_clusters.csv"),
              row.names=FALSE)
  }
  
  if (length(keep_names) < n_domains){
    stop("only ", length(keep_names), " clusters survive the never-present filter, ",
         "fewer than the ", n_domains, " domains asked for; cutree cannot make ",
         "more groups than it has leaves")
  }
  
  count_clusters_hard = count_clusters_hard[, c(1, keep_index + 1)]
  
  for (cur_region in region_ids){
    embed_list[[cur_region]] = embed_list[[cur_region]][keep_index, keep_index, drop=FALSE]
    coord_list[[cur_region]] = coord_list[[cur_region]][keep_index, keep_index, drop=FALSE]
  }
  
  n_kmeans_clusters = length(keep_names)
  
  count_list = list()
  
  for (c_name in colnames(count_clusters_hard)[2:(n_kmeans_clusters+1)]){
    count_list[[c_name]] = count_clusters_hard[[c_name]]
  }
  
  summary_embed_matrx = matrix(NA, ncol=n_kmeans_clusters, nrow=n_kmeans_clusters)
  summary_coord_matrx = matrix(NA, ncol=n_kmeans_clusters, nrow=n_kmeans_clusters)
  
  coexist_count = matrix(NA, ncol=n_kmeans_clusters, nrow=n_kmeans_clusters)
  
  col_cluster1 = NULL
  col_cluster2 = NULL
  col_n_cluster1 = NULL
  col_n_cluster2 = NULL
  
  coexist_pairs_count = NULL
  coexist_pairs_pos_pvalue = NULL
  coexist_pairs_neg_pvalue = NULL
  coexist_pairs_two_pvalue = NULL
  expected_overlap = NULL
  
  col_embed_dist = NULL
  col_coord_dist = NULL
  
  for (i in 1:(n_kmeans_clusters-1)){
    for (j in (i+1):n_kmeans_clusters){
      
      c_name_i = colnames(count_clusters_hard)[2:(n_kmeans_clusters+1)][i]
      c_name_j = colnames(count_clusters_hard)[2:(n_kmeans_clusters+1)][j]
      tf_i = as.integer(count_list[[c_name_i]]>=n_cells_threshold)
      tf_j = as.integer(count_list[[c_name_j]]>=n_cells_threshold)
      exists_i = which(count_list[[c_name_i]]>=n_cells_threshold)
      exists_j = which(count_list[[c_name_j]]>=n_cells_threshold) 
      exists_ij_both = intersect(exists_i, exists_j)
      exists_both_images = count_clusters_hard$X[exists_ij_both]
      
      coexist_count[i, j] = length(exists_ij_both)
      coexist_count[j, i] = length(exists_ij_both)
      coexist_pairs_count = c(coexist_pairs_count, length(exists_ij_both))
      
      if ((length(unique(tf_i))==2)&(length(unique(tf_j))==2)){
        pos_p = fisher.test(tf_i,
                            tf_j,
                            alternative = "greater")$p.value
  
        neg_p = fisher.test(tf_i,
                            tf_j,
                            alternative = "less")$p.value
  
        two_p = fisher.test(tf_i,
                             tf_j,
                             alternative = "two.sided")$p.value
      }else{
        pos_p = NA
        neg_p = NA
        two_p = NA
      }
      
      col_cluster1 = c(col_cluster1, c_name_i)
      col_cluster2 = c(col_cluster2, c_name_j)
      col_n_cluster1 = c(col_n_cluster1, sum(tf_i))
      col_n_cluster2 = c(col_n_cluster2, sum(tf_j))
      
      
      coexist_pairs_pos_pvalue = c(coexist_pairs_pos_pvalue, pos_p)
      coexist_pairs_neg_pvalue = c(coexist_pairs_neg_pvalue, neg_p)
      coexist_pairs_two_pvalue = c(coexist_pairs_two_pvalue, two_p)
      
      expected_overlap = c(expected_overlap, sum(tf_i)*mean(tf_j))
      
      embed_dists_ij = NULL
      coord_dists_ij = NULL
      
  
      for (image_name in exists_both_images){
          embed_dists_ij = c(embed_dists_ij, as.numeric(embed_list[[image_name]][[i, j]]))
          coord_dists_ij = c(coord_dists_ij, as.numeric(coord_list[[image_name]][[i, j]]))   
      }
    
      if (length(exists_both_images)>=n_images_threshold){
        summary_embed_matrx[i, j] = median(embed_dists_ij, na.rm=TRUE)
        summary_coord_matrx[i, j] = median(coord_dists_ij, na.rm=TRUE)
        summary_embed_matrx[j, i] = median(embed_dists_ij, na.rm=TRUE)
        summary_coord_matrx[j, i] = median(coord_dists_ij, na.rm=TRUE)
      }
      
      if (length(exists_both_images)==0){
        col_embed_dist = c(col_embed_dist, NA)
        col_coord_dist = c(col_coord_dist, NA)  
      }else{
        col_embed_dist = c(col_embed_dist, median(embed_dists_ij, na.rm=TRUE))
        col_coord_dist = c(col_coord_dist, median(coord_dists_ij, na.rm=TRUE))        
      }

    }
  }
    
  sum(!is.na(summary_embed_matrx))
  sum(!is.na(summary_coord_matrx))
  
  c_embed = c(summary_embed_matrx)
  c_coord = c(summary_coord_matrx)
  
  dim(count_clusters_hard)
  
  print(sum(coexist_pairs_count >= 0.9*nrow(count_clusters_hard)))
  print(sum(coexist_pairs_count >= 0.8*nrow(count_clusters_hard)))
  print(sum(coexist_pairs_count >= 0.7*nrow(count_clusters_hard)))
  print(sum(coexist_pairs_count >= 0.6*nrow(count_clusters_hard)))
  print(sum(coexist_pairs_count >= n_images_threshold))
  
  df_cooccur = data.frame(cluster_i = col_cluster1, 
                          cluster_j = col_cluster2, 
                          n_exist_i = col_n_cluster1,
                          n_exist_j = col_n_cluster2, 
                          coexist_count = coexist_pairs_count, 
                          expected_overlap = expected_overlap,
                          pos_pval = coexist_pairs_pos_pvalue, 
                          neg_pval = coexist_pairs_neg_pvalue, 
                          two_pval = coexist_pairs_two_pvalue, 
                          embed_dist = col_embed_dist, 
                          coord_dist = col_coord_dist)
  
  print(head(df_cooccur))
  
  embed_dist_median = median(c_embed, na.rm=TRUE)
  coord_dist_median = median(c_coord, na.rm=TRUE)
  
  print(embed_dist_median)
  print(coord_dist_median)
  
  scale_factor = (coord_dist_median/embed_dist_median)
  
  df_to_fill = df_cooccur[which(df_cooccur$coexist_count<n_images_threshold),]
  dim(df_to_fill)
  summary(df_to_fill$coexist_count)
  
  weighted_matrix = ((summary_embed_matrx*scale_factor)+summary_coord_matrx)/2
  
  for (i in 1:n_kmeans_clusters){
    weighted_matrix[i,i] = 0
  }
  
  max_value = max(weighted_matrix, na.rm=TRUE)
  print(max_value)
  
  get_interpolated_value <- function(i, j, coexist_count, weighted_matrix, max_value, n_images_threshold){
    stopifnot(i != j)
    if (!is.na(weighted_matrix[i, j])){
      return (weighted_matrix[i, j])
    }else{
      n_coexist = coexist_count[i, j]
      stopifnot(!is.na(n_coexist))
      stopifnot(n_coexist < n_images_threshold)
      interpolated_value = max_value + max_value * (n_images_threshold - n_coexist)/n_images_threshold
      return (interpolated_value)
    }
  }
  
  for (i in 1:(n_kmeans_clusters-1)){
    for (j in (i+1):n_kmeans_clusters){
      weighted_matrix[i, j] = get_interpolated_value(i, j, coexist_count, weighted_matrix, max_value, n_images_threshold)
      weighted_matrix[j, i] = weighted_matrix[i, j]
    }
  }
  
  stopifnot(sum(is.na(weighted_matrix))==0)
  print(max(weighted_matrix))
  
  row.names(weighted_matrix) = colnames(count_clusters_hard)[2:(n_kmeans_clusters+1)]
  colnames(weighted_matrix) = row.names(weighted_matrix)
  
  saveRDS(weighted_matrix, file = file.path(domain_dir, "weighted_dist_matrix.rds"))
  
  weighted_dist = as.dist(weighted_matrix)
  
  output_figure = file.path(domain_dir, "weighted_dist_hclust.pdf")
  pdf(file = output_figure, 
      width = max(0.8*n_domains, 6), height = 4)
  hc = hclust(weighted_dist, method = "complete", members = NULL)
  plot(hc)
  dev.off()  
  
  clusterCut <- cutree(hc, k = n_domains)
  print(table(clusterCut))
  
  df_clusterCut = as.data.frame(clusterCut)
  colnames(df_clusterCut) = c("domain")
  df_clusterCut$cluster = row.names(df_clusterCut)
  
  write.csv(df_clusterCut, 
            file.path(domain_dir, "domain_kmeans_clusters_membership.csv"), 
            row.names=FALSE)

  df_cluster$kmeans_cluster_string = paste0("cluster_", 
                                            as.character(df_cluster$kmeans_cluster))
  
  df_domain_matched = df_clusterCut[match(df_cluster$kmeans_cluster_string, 
                                          df_clusterCut$cluster),]
  
  matched = !is.na(df_domain_matched$cluster)
  stopifnot(all(df_cluster$kmeans_cluster_string[matched] ==
                  df_domain_matched$cluster[matched]))
  stopifnot(all(df_cluster$kmeans_cluster_string[!matched] %in% drop_names))
  
  df_cluster$domain = df_domain_matched$domain
  
  stopifnot(all(!is.na(df_cluster$domain)))
  
  cat("domain assigned to all ", nrow(df_cluster), " cells\n", sep="")
  
  df_output = df_cluster[, c("CELL_ID", "domain")]
  
  write.csv(df_output, 
            file.path(domain_dir, "domain_annotation_for_cells.csv"), 
            row.names=FALSE)
  
}

