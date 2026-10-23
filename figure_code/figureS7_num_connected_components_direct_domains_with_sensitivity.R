library(data.table)
library(ggplot2)
library(grid)

options(bitmapType = "cairo")
theme_set(theme_classic())

figure_dir = file.path("../revision_figures",
                       "figure5_num_connected_components_direct_domains_with_sensitivity")
if (!dir.exists(figure_dir)){
  dir.create(figure_dir, recursive = TRUE)
}

figure_width = 8.5
figure_height = 9.5
base_font_size = 13
strip_font_size = 13

revision_root = "../revision/analysis/figures"

method_files = list(
  "K-means" = file.path(revision_root, "n_connected_components_raw_kmeans.csv"),
  "Spatial-LDA" = "../SpatialLDA_copy/results/step8_avg_n_connected_components_spalda_radius_27_d_0.25.csv",
  "SpatialTopic" = "../SpaTopic_copy/results/step7_avg_n_connected_components_spatopic_radius_62_sigma_8.csv",
  "Concordia-Basic" = file.path(revision_root,
                                "n_connected_components_rs_ensemble_scaled_direct_domains_basic.csv"),
  "Concordia-Local" = file.path(revision_root,
                                "n_connected_components_rs_ensemble_scaled_direct_domains_local.csv"),
  "Concordia (default)" = file.path(revision_root,
                                    "n_connected_components_rs_ensemble_scaled_direct_domains.csv")
)

method_colors = setNames(c("#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"),
                         names(method_files))

settings_csv = file.path(revision_root, "n_connected_components_sensitivity_settings.csv")

reference_label = "Concordia (default)"

group_levels = c("Methods", "Sensitivity settings", "Number of domains")
group_colors = c("Sensitivity settings" = "grey30", "Number of domains" = "#2c7fb8")

missing_files = c(unlist(method_files), settings_csv)
missing_files = missing_files[!file.exists(missing_files)]
if (length(missing_files) > 0){
  stop("missing:\n  ", paste(missing_files, collapse = "\n  "))
}


box_rows = list()
region_reference = NULL

for (cur_method in names(method_files)){

  df_cur = fread(method_files[[cur_method]],
                 select = c("region_id", "avg_n_connected_components"))

  stopifnot(!any(duplicated(df_cur$region_id)))

  if (is.null(region_reference)){
    region_reference = sort(df_cur$region_id)
  }else{
    stopifnot(identical(sort(df_cur$region_id), region_reference))
  }

  box_rows[[length(box_rows) + 1]] = data.table(
    group = "Methods", label = cur_method, color = method_colors[[cur_method]],
    region_id = df_cur$region_id, value = df_cur$avg_n_connected_components)
}


df_settings = fread(settings_csv)

stopifnot(all(df_settings[setting == "permutate", setting_label] ==
                "permuted data, permutate"))
df_settings[setting == "permutate", setting_label := "permuted data, permutate10pct"]

stopifnot(identical(sort(unique(df_settings$region_id)), region_reference))

df_default = merge(df_settings[setting == "default", .(region_id, avg_n_connected_components)],
                   box_rows[[length(box_rows)]][, .(region_id, value)], by = "region_id")
stopifnot(nrow(df_default) == length(region_reference))
stopifnot(isTRUE(all.equal(df_default$avg_n_connected_components, df_default$value)))

setting_order = unique(df_settings[setting != "default", .(group, setting, setting_label)])

for (setting_index in seq_len(nrow(setting_order))){

  cur = setting_order[setting_index]

  cur_group = if (cur$group == "sensitivity") "Sensitivity settings" else "Number of domains"

  cur_label = if (cur_group == "Sensitivity settings"){
    paste0(toupper(substr(cur$setting_label, 1, 1)),
           substr(cur$setting_label, 2, nchar(cur$setting_label)))
  }else{
    cur$setting_label
  }

  df_cur = df_settings[setting == cur$setting]
  stopifnot(!any(duplicated(df_cur$region_id)))

  box_rows[[length(box_rows) + 1]] = data.table(
    group = cur_group, label = cur_label, color = group_colors[[cur_group]],
    region_id = df_cur$region_id, value = df_cur$avg_n_connected_components)
}

df_box = rbindlist(box_rows)

label_order = unique(df_box$label)
stopifnot(!any(duplicated(label_order)))

n_missing = df_box[, .(n_missing = sum(!is.finite(value))), by = label][n_missing > 0]
if (nrow(n_missing) > 0){
  cat("images without a value, left out of their box:\n")
  print(n_missing, row.names = FALSE)
}

df_box = df_box[is.finite(value)]
stopifnot(all(df_box$value > 0))

df_box[, log10_value := log10(value)]

df_box[, label := factor(label, levels = rev(label_order))]
df_box[, group := factor(group, levels = group_levels)]

fwrite(df_box[, .(group, label, region_id, avg_n_connected_components = value,
                  log10_avg_n_connected_components = log10_value)],
       file.path(figure_dir, "n_connected_components_by_image.csv"))

df_summary = df_box[, .(n_images = .N, median = median(value),
                        q25 = quantile(value, 0.25), q75 = quantile(value, 0.75)),
                    by = .(group, label)]

reference_median = df_summary[label == reference_label, median]

options(width = 200)
cat("\nmedian average number of connected components per domain; reference ",
    format(round(reference_median, 2), nsmall = 2), "\n", sep = "")
print(df_summary[order(match(as.character(label), label_order)),
                 .(group, label, n_images, median = round(median, 2),
                   q25 = round(q25, 2), q75 = round(q75, 2))],
      row.names = FALSE)


color_of = setNames(unique(df_box[, .(label, color)])$color,
                    as.character(unique(df_box[, .(label, color)])$label))

p_box = ggplot(df_box, aes(x = log10_value, y = label, color = label)) +
  geom_vline(xintercept = log10(reference_median), linetype = "dashed",
             colour = method_colors[[reference_label]], linewidth = 0.4) +
  geom_boxplot(outlier.size = 0.4, outlier.alpha = 0.4, fill = NA,
               width = 0.6, linewidth = 0.5) +
  scale_color_manual(values = color_of, guide = "none") +
  scale_x_continuous(breaks = seq(0, 3, by = 0.5)) +
  facet_grid(group ~ ., scales = "free_y", space = "free_y") +
  labs(x = expression(log[10] * "(average number of connected components per domain)"),
       y = NULL) +
  theme_classic(base_size = base_font_size) +
  theme(strip.text.y = element_text(size = strip_font_size, angle = -90),
        strip.background = element_rect(colour = NA, fill = "grey92"),
        panel.spacing = unit(8, "pt"),
        axis.text.x = element_text(margin = margin(t = 5)))

figure_file = file.path(figure_dir, "boxplots.pdf")

while (!is.null(dev.list())) dev.off()

pdf(figure_file, width = figure_width, height = figure_height)
print(p_box)
invisible(dev.off())

cat("\nsaved ", figure_file, "\n", sep = "")
cat("saved ", file.path(figure_dir, "n_connected_components_by_image.csv"), "\n", sep = "")
