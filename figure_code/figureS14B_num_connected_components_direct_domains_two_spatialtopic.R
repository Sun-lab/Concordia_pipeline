library(tidyr)
library(dplyr)
library(ggplot2)
library(data.table)

options(bitmapType = "cairo")

theme_set(theme_classic())

figure_output_dir = "../revision_figures"

if (!dir.exists(figure_output_dir)){
  dir.create(figure_output_dir, recursive = TRUE)
}

example_region_id = "86_C6_10"

revision_root = "../revision/analysis/figures"

file_list = list()

file_list[["K-means"]] = file.path(revision_root,
                                   "n_connected_components_raw_kmeans.csv")
file_list[["Spatial-LDA"]] = "../SpatialLDA_copy/results/step8_avg_n_connected_components_spalda_radius_27_d_0.25.csv"
file_list[["SpatialTopic"]] ="../SpaTopic_copy/results/step7_avg_n_connected_components_spatopic_radius_62_sigma_8.csv"
file_list[["SpatialTopic (radius=27)"]] ="../SpaTopic_copy/results/step7_avg_n_connected_components_spatopic_radius_27_sigma_5.csv"
file_list[["Concordia-Basic"]] = file.path(revision_root,
                                           "n_connected_components_rs_ensemble_scaled_direct_domains_basic.csv")
file_list[["Concordia-Local"]] = file.path(revision_root,
                                           "n_connected_components_rs_ensemble_scaled_direct_domains_local.csv")
file_list[["Concordia"]] = file.path(revision_root,
                                     "n_connected_components_rs_ensemble_scaled_direct_domains.csv")

method_levels = names(file_list)

method_colors <- setNames(c("#E69F00", "#56B4E9", "#009E73", "#F0E442",
                            "#0072B2", "#D55E00", "#CC79A7"),
                          method_levels)



missing_files = unlist(file_list)[!file.exists(unlist(file_list))]

if (length(missing_files) > 0){
  stop("missing ", length(missing_files), " of ", length(file_list),
       " method files:\n  ", paste(missing_files, collapse = "\n  "))
}

df = NULL
region_reference = NULL

for (name in method_levels){

  df_tmp = fread(file_list[[name]],
                 select = c("region_id", "avg_n_connected_components"))

  cur_regions = sort(df_tmp$region_id)

  if (is.null(region_reference)){
    region_reference = cur_regions
    stopifnot(!any(duplicated(region_reference)))
  }else{
    stopifnot(identical(cur_regions, region_reference))
  }

  df_tmp$method = name
  df = rbind(df, df_tmp)

  cat(sprintf("%-24s %4d images, median %6.2f\n", name, nrow(df_tmp),
              median(df_tmp$avg_n_connected_components, na.rm = TRUE)))
}

df = as.data.frame(df)

cat("\nall", length(file_list), "cover the same", length(region_reference), "images\n")

df$method = factor(df$method, levels = method_levels)

output_csv = file.path(figure_output_dir,
                       "n_connected_components_seven_methods_two_spatialtopic.csv")

fwrite(df, output_csv)



stopifnot(all(is.finite(df$avg_n_connected_components)))
stopifnot(all(df$avg_n_connected_components > 0))

df$log10_avg_n_connected_components = log10(df$avg_n_connected_components)

cat("\nboxplots are on log10 of the metric, ",
    sprintf("%.3f to %.3f", min(df$log10_avg_n_connected_components),
            max(df$log10_avg_n_connected_components)),
    " (raw ", sprintf("%.2f to %.2f", min(df$avg_n_connected_components),
                      max(df$avg_n_connected_components)), ")\n", sep = "")

for (name in method_levels){
  cur = df$log10_avg_n_connected_components[df$method == name]
  cat(sprintf("%-24s median log10 %6.3f  = %6.2f components\n",
              name, median(cur), 10 ^ median(cur)))
}



df_one = df[which(df$region_id == example_region_id), ]
stopifnot(nrow(df_one) == length(method_levels))

cat("\n", example_region_id, ":\n", sep = "")
print(df_one, row.names = FALSE)

df_one$method = factor(df_one$method, levels = method_levels)



p0 <- ggplot(df_one, aes(x = method, y = avg_n_connected_components, fill = method)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = method_colors) +
  guides(fill = guide_legend(title = NULL)) +
  theme(text = element_text(size = 14),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        legend.text  = element_text(size = 14),
        legend.title = element_text(size = 14),
        legend.key.size = grid::unit(0.6, "cm"))

p1 <- ggplot(df, aes(x = method, y = log10_avg_n_connected_components,
                     fill = method)) +
  geom_boxplot() +
  scale_fill_manual(values = method_colors) +
  guides(fill = guide_legend(title = NULL)) +
  theme(text = element_text(size = 14),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        legend.text  = element_text(size = 14),
        legend.title = element_text(size = 14),
        legend.key.size = grid::unit(0.6, "cm"))


figure_width_bar = 5
figure_width_box = 5.5
figure_height = 2.6

figure_file_bar = file.path(figure_output_dir,
                            paste0("figure5_connected_components_", example_region_id,
                                   "_direct_domains_two_spatialtopic.pdf"))

pdf(file = figure_file_bar, width = figure_width_bar, height = figure_height)
print(p0)
dev.off()

cat("\nsaved", figure_file_bar, "\n")

figure_file_box = file.path(figure_output_dir,
                            "figure5_connected_components_all_direct_domains_two_spatialtopic.pdf")

pdf(file = figure_file_box, width = figure_width_box, height = figure_height)
print(p1)
dev.off()

cat("saved", figure_file_box, "\n")
cat("saved", output_csv, "\n")


gc()
sessionInfo()
