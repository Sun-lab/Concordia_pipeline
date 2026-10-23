library(ggplot2)
library(data.table)

options(bitmapType = "cairo")

theme_set(theme_classic())

figure_output_dir = "../revision_figures"

if (!dir.exists(figure_output_dir)){
  dir.create(figure_output_dir, recursive = TRUE)
}

revision_root = "../revision/analysis/figures"

file_list = list()

file_list[["Concordia"]] = file.path(revision_root,
                                     "n_connected_components_lermi_lermi_trained_direct_domains.csv")
file_list[["Concordia, transferred"]] = file.path(revision_root,
                                                  "n_connected_components_lermi_cords_transferred_direct_domains.csv")

method_levels = names(file_list)

method_colors <- setNames(c("#CC79A7", "#0072B2"), method_levels)


missing_files = unlist(file_list)[!file.exists(unlist(file_list))]

if (length(missing_files) > 0){
  stop("missing ", length(missing_files), " of ", length(file_list),
       " files:\n  ", paste(missing_files, collapse = "\n  "))
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
}

df = as.data.frame(df)

df$method = factor(df$method, levels = method_levels)

stopifnot(all(is.finite(df$avg_n_connected_components)))
stopifnot(all(df$avg_n_connected_components > 0))

df$log10_avg_n_connected_components = log10(df$avg_n_connected_components)

df_wide = dcast(as.data.table(df), region_id ~ method,
                value.var = "avg_n_connected_components")
n_lower = sum(df_wide[[method_levels[2]]] < df_wide[[method_levels[1]]])

cat(method_levels[2], "is lower than", method_levels[1], "in", n_lower, "of",
    nrow(df_wide), "images\n")


p1 <- ggplot(df, aes(x = method, y = log10_avg_n_connected_components)) +
  geom_boxplot(aes(fill = method), outlier.shape = NA, alpha = 0.5, width = 0.6) +
  geom_line(aes(group = region_id), color = "grey45", linewidth = 0.35) +
  geom_point(aes(color = method), size = 1.4) +
  scale_fill_manual(values = method_colors) +
  scale_color_manual(values = method_colors) +
  guides(fill = guide_legend(title = NULL, ncol = 1), color = "none") +
  theme(text = element_text(size = 14),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        legend.position = "bottom",
        legend.text  = element_text(size = 14),
        legend.title = element_text(size = 14),
        legend.key.size = grid::unit(0.6, "cm"))

figure_file = file.path(figure_output_dir,
                        "figure5_connected_components_lermi_trained_vs_transferred_paired.pdf")

pdf(file = figure_file, width = 2.6, height = 3.2)
print(p1)
dev.off()

cat("saved", figure_file, "\n")

sessionInfo()
