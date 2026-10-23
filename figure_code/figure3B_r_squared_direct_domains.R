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

published_ylim = c(0.45, 1.0)
clip_floor = 0.5

kmeans_file = paste0("../results/cords_2024/extended/",
                     "cords_d20_comp2nd_both_extended_sparse_mincutpool_prop_unsupervised_gat2/",
                     "cords_d20_comp2nd_both_sparse_mincutpool_prop_unsupervised_gat2_add_gpu_clusters_40_",
                     "gcn_2_1_o1_1.0_o2_1.0_batch_64_lr_0001_epoch_1000/",
                     "r2_domain_and_image_all_methods.csv")

revision_root = "../revision/analysis/figures/anova_combined_r2_direct_domains"

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

df_kmeans = df_all[which(df_all$method == "raw_kmeans"), ]
stopifnot(nrow(df_kmeans) > 0)
df_kmeans$method = "K-means"

cat("K-means dimensions:", nrow(df_kmeans), "\n")



revision_files = file.path(revision_root, names(method_labels), "r2_domain_and_both.csv")
names(revision_files) = names(method_labels)

missing_files = revision_files[!file.exists(revision_files)]

if (length(missing_files) > 0){
  stop("missing ", length(missing_files), " of ", length(revision_files),
       " method tables, so the panel would be incomplete:\n  ",
       paste(missing_files, collapse = "\n  "),
       "\nrun revision/analysis/anova_combined_r2_direct_domains.sh first")
}

df_revision = NULL

for (cur_method in names(method_labels)){

  cur = read.csv(revision_files[[cur_method]], header = TRUE)
  stopifnot(all(cur$method == cur_method))

  cur$method = unname(method_labels[cur_method])

  cat(unname(method_labels[cur_method]), "dimensions:", nrow(cur), "\n")

  df_revision = rbind(df_revision, cur)
}



shared_cols = intersect(colnames(df_kmeans), colnames(df_revision))
df_r2 = rbind(df_kmeans[, shared_cols], df_revision[, shared_cols])

df_r2$method = factor(df_r2$method, levels = panel_levels)
stopifnot(!any(is.na(df_r2$method)))

write.csv(df_r2, file.path(figure_dir, "r2_by_dimension_all_methods.csv"),
          row.names = FALSE)

cat("\n==== r.squared by domain ====\n")
print(df_r2 %>%
        group_by(method) %>%
        summarise(n_dim = n(), median = median(r.squared),
                  min = min(r.squared), max = max(r.squared), .groups = "drop") %>%
        as.data.frame(), row.names = FALSE)



clipped_points = df_r2 %>% filter(r.squared < clip_floor)

cat("\nPoints below the published clip floor of", clip_floor, ":",
    nrow(clipped_points), "\n")
if (nrow(clipped_points) > 0){
  print(clipped_points %>% count(method, name = "n_below") %>% as.data.frame(),
        row.names = FALSE)
}

use_published_window = (nrow(clipped_points) == 0)

if (use_published_window){
  y_values = pmax(df_r2$r.squared, clip_floor)
  y_limits = published_ylim
  cat("every point is above the floor, so the published window is used\n")
}else{
  y_values = df_r2$r.squared
  y_pad = 0.03 * (max(df_r2$r.squared) - min(df_r2$r.squared))
  y_limits = c(min(df_r2$r.squared) - y_pad, max(df_r2$r.squared) + y_pad)
  cat("the axis is widened to", sprintf("%.2f", y_limits[1]), "-",
      sprintf("%.2f", y_limits[2]), "so nothing is clipped\n")
}

df_r2$y_plot = y_values

p1 = ggplot(df_r2, aes(x = method, y = r.squared, color = method)) +
  geom_boxplot(outlier.shape = NA) +
  geom_jitter(aes(y = y_plot), width = 0.2, alpha = 0.6, size = 1, shape = 16) +
  theme(axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank()) +
  coord_cartesian(ylim = y_limits)

figure_file = file.path(figure_dir, "figure3b_r2_direct_domains.pdf")

pdf(file = figure_file, width = 4.6, height = 1.9)
print(p1)
dev.off()

cat("\nsaved", figure_file, "\n")
cat("saved", file.path(figure_dir, "r2_by_dimension_all_methods.csv"), "\n")


gc()
sessionInfo()
