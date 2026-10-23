save_summary_dist_matrices <- function(domain_dir, region_ids, n_kmeans_clusters,
                                n_cells_threshold, n_images_threshold){

  embed_list = list()
  coord_list = list()
  
  for (cur_region in region_ids){
    
    df_embed_cur = read.csv(file.path(domain_dir, "embedding_dist_in_image", 
                                      sprintf("embedding_dist_%s.csv", cur_region)), 
                            header=FALSE)
    embed_list[[cur_region]] = as.matrix(df_embed_cur)
    
    df_coord_cur = read.csv(file.path(domain_dir, "coord_dist_in_image", 
                                      sprintf("coord_dist_%s.csv", cur_region)), 
                            header=FALSE)  
    coord_list[[cur_region]] = as.matrix(df_coord_cur)
    
  }
  
  count_clusters_hard = read.csv(file.path(domain_dir, paste0("cluster_hard_count_domains.csv")), 
                                 header=TRUE)
  
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
  
  cluster_names = paste0("domain_", as.character(1:n_kmeans_clusters))

  row.names(summary_embed_matrx) = cluster_names
  colnames(summary_embed_matrx) = cluster_names
  row.names(summary_coord_matrx) = cluster_names
  colnames(summary_coord_matrx) = cluster_names

  n_pairs = n_kmeans_clusters*(n_kmeans_clusters-1)/2
  n_missing = sum(is.na(summary_embed_matrx[upper.tri(summary_embed_matrx)]))

  cat("\npairs with a median: ", n_pairs-n_missing, " of ", n_pairs, "; ",
      n_missing, " left as NA, co-occurring in fewer than ", n_images_threshold,
      " images\n", sep="")

  saveRDS(summary_embed_matrx,
          file = file.path(domain_dir, "summary_embed_dist_matrix.rds"))
  saveRDS(summary_coord_matrx,
          file = file.path(domain_dir, "summary_coord_dist_matrix.rds"))

  write.csv(summary_embed_matrx,
            file = file.path(domain_dir, "summary_embed_dist_matrix.csv"),
            row.names=TRUE)
  write.csv(summary_coord_matrx,
            file = file.path(domain_dir, "summary_coord_dist_matrix.csv"),
            row.names=TRUE)

  write.csv(df_cooccur,
            file = file.path(domain_dir, "summary_dist_cooccurrence.csv"),
            row.names=FALSE)

}
