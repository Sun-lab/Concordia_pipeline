get_n_cells_in_clusters <- function(raw_dir, dir, region_ids, n_kmeans_clusters){
  
  cluster_file = file.path(dir, "kmeans_cluster.csv")
  df_cluster = read.csv(cluster_file, header=TRUE)
  dim(df_cluster)  
  head(df_cluster)
  
  count_clusters_hard = NULL
  
  for (cur_region in region_ids){
    
    df_cur = read.csv(paste0(raw_dir, "/", cur_region, ".csv"), 
                      header=TRUE)
    
    df_cluster_matched = df_cluster[match(df_cur$CELL_ID, 
                                          df_cluster$CELL_ID), ]
    stopifnot(all(df_cur$CELL_ID==df_cluster_matched$CELL_ID))    
    
    df_cur$cluster = paste0("cluster_", as.character(df_cluster_matched$kmeans_cluster))
    df_cur$cluster = factor(df_cur$cluster, 
                            levels=paste0("cluster_", as.character(0:(n_kmeans_clusters-1))))
    
    cur_table = as.data.frame(table(df_cur$cluster))
    
    stopifnot(all(cur_table$Var1==paste0("cluster_", as.character(0:(n_kmeans_clusters-1)))))
    
    cluster_counts_hard = cur_table$Freq
    
    count_clusters_hard = rbind(count_clusters_hard, cluster_counts_hard)
    
  }
  
  dim(count_clusters_hard)
  row.names(count_clusters_hard) =region_ids
  colnames(count_clusters_hard) = paste0("cluster_", as.character(0:(n_kmeans_clusters-1)))
  
  write.csv(count_clusters_hard, 
            file = file.path(dir, paste0("cluster_hard_count_kmeans.csv")), 
            row.names=TRUE)

}

