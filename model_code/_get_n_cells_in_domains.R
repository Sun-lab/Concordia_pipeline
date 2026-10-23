get_n_cells_in_domains <- function(raw_dir, domain_dir, region_ids, n_domains){
  
  domain_file = file.path(domain_dir, "domain_annotation_for_cells.csv")
  df_cluster = read.csv(domain_file, header=TRUE)
  stopifnot(all(df_cluster$domain %in% 1:n_domains))
  dim(df_cluster)  
  head(df_cluster)
  
  count_clusters_hard = NULL
  
  for (cur_region in region_ids){
    
    df_cur = read.csv(paste0(raw_dir, "/", cur_region, ".csv"), 
                      header=TRUE)
    
    df_cluster_matched = df_cluster[match(df_cur$CELL_ID, 
                                          df_cluster$CELL_ID), ]
    stopifnot(all(df_cur$CELL_ID==df_cluster_matched$CELL_ID))    
    
    df_cur$cluster = paste0("domain_", as.character(df_cluster_matched$domain))
    df_cur$cluster = factor(df_cur$cluster, 
                            levels=paste0("domain_", as.character(1:n_domains)))
    
    cur_table = as.data.frame(table(df_cur$cluster))
    
    stopifnot(all(cur_table$Var1==paste0("domain_", as.character(1:n_domains))))
    
    cluster_counts_hard = cur_table$Freq
    
    count_clusters_hard = rbind(count_clusters_hard, cluster_counts_hard)
    
  }
  
  dim(count_clusters_hard)
  row.names(count_clusters_hard) =region_ids
  colnames(count_clusters_hard) = paste0("domain_", as.character(1:n_domains))
  
  write.csv(count_clusters_hard, 
            file = file.path(domain_dir, paste0("cluster_hard_count_domains.csv")), 
            row.names=TRUE)

}

