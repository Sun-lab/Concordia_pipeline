library(tidyr)
library(dplyr)
library(data.table)
library(ggplot2)

theme_set(theme_classic())
options(bitmapType = "cairo")

figure_dir = "../revision_figures"
if (!dir.exists(figure_dir)){
  dir.create(figure_dir, recursive = TRUE)
}

kmeans_file = paste0("../results/cords_2024/extended/",
                     "cords_d20_comp2nd_both_extended_sparse_mincutpool_prop_unsupervised_gat2/",
                     "cords_d20_comp2nd_both_sparse_mincutpool_prop_unsupervised_gat2_add_gpu_clusters_40_",
                     "gcn_2_1_o1_1.0_o2_1.0_batch_64_lr_0001_epoch_1000/",
                     "morans_i_row_being_one_dim.csv")

revision_root = "../revision/analysis/figures/morans_i_in_image_direct_domains"

method_labels = c("spatial_lda" = "Spatial-LDA",
                  "spatopic" = "SpatialTopic",
                  "basic" = "Concordia-Basic",
                  "local" = "Concordia-Local",
                  "extended" = "Concordia")

panel_levels = c("K-means", unname(method_labels))



if (!file.exists(kmeans_file)){
  stop("missing ", kmeans_file)
}

df_all = read.csv(kmeans_file, header = TRUE)

df_kmeans = df_all[which(df_all$setting == "prop"), ]
stopifnot(nrow(df_kmeans) > 0)
df_kmeans$setting = "K-means"

cat("K-means dimensions:", nrow(df_kmeans), "\n")



revision_files = file.path(revision_root, names(method_labels),
                           "morans_i_median_by_dim.csv")
names(revision_files) = names(method_labels)

missing_files = revision_files[!file.exists(revision_files)]

if (length(missing_files) > 0){
  stop("missing ", length(missing_files), " of ", length(revision_files),
       " method tables, so the panel would be incomplete:\n  ",
       paste(missing_files, collapse = "\n  "),
       "\nrun revision/analysis/morans_i_in_image_direct_domains.sh first")
}

df_revision = NULL

for (cur_method in names(method_labels)){

  cur = read.csv(revision_files[[cur_method]], header = TRUE)
  stopifnot(all(cur$setting == cur_method))

  cur$setting = unname(method_labels[cur_method])

  cat(unname(method_labels[cur_method]), "dimensions:", nrow(cur), "\n")

  df_revision = rbind(df_revision, cur)
}



shared_cols = intersect(colnames(df_kmeans), colnames(df_revision))
df_morans_i = rbind(df_kmeans[, shared_cols], df_revision[, shared_cols])

df_morans_i$method = factor(df_morans_i$setting, levels = panel_levels)
stopifnot(!any(is.na(df_morans_i$method)))

write.csv(df_morans_i, file.path(figure_dir, "morans_i_by_dimension_all_methods.csv"),
          row.names = FALSE)

cat("\n==== Moran's I, median across images, by dimension ====\n")
print(df_morans_i %>%
        group_by(method) %>%
        summarise(n_dim = n(), median = median(morans_i),
                  min = min(morans_i), max = max(morans_i), .groups = "drop") %>%
        as.data.frame(), row.names = FALSE)



p1 = ggplot(df_morans_i, aes(x = method, y = morans_i, color = method)) +
  geom_boxplot(outlier.shape = NA) +
  geom_jitter(width = 0.2, alpha = 0.6, size = 1, shape = 16) +
  labs(y = "Moran's I") +
  theme(axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank())

figure_file = file.path(figure_dir, "figure3a_morans_i_direct_domains.pdf")

pdf(file = figure_file, width = 4.6, height = 1.9)
print(p1)
dev.off()

cat("\nsaved", figure_file, "\n")
cat("saved", file.path(figure_dir, "morans_i_by_dimension_all_methods.csv"), "\n")


gc()
sessionInfo()
