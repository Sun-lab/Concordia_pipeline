library(data.table)
library(ggplot2)
library(grid)

theme_set(theme_classic())
options(bitmapType = "cairo")

figure_dir = file.path("../revision_figures",
                       "figure5_f_lin_boxplots_high_mean_prop_t_with_sensitivity")
if (!dir.exists(figure_dir)){
  dir.create(figure_dir, recursive = TRUE)
}

min_mean_prop_t = 0.85

figure_width = 8.5
figure_height = 10
base_font_size = 13
strip_font_size = 13
count_font_size = 10

count_x = 1.04
x_upper = 1.17

metric_dir = "../revision/analysis/figures"

fraction_csv = file.path(metric_dir,
                         "tumor_neighbor_tumor_fraction_sampled_images_random_sample.csv")
metric_csv = file.path(metric_dir,
                       "local_pca_dimension_cell_min_all_methods_random_sample.csv")
match_metric_csv = file.path(metric_dir,
                             "local_pca_dimension_cell_min_default_domain10_matches_random_sample.csv")

group_levels = c("Methods", "Sensitivity settings", "Number of domains")

method_targets = list(
  list(label = "K-means, domain 6", domain = 6, metric_method = "raw_kmeans",
       color = "#1f77b4"),
  list(label = "Concordia-Basic, domain 7", domain = 7, metric_method = "direct_basic",
       color = "#9467bd"),
  list(label = "Concordia-Local, domain 6", domain = 6, metric_method = "direct_local",
       color = "#8c564b"),
  list(label = "Concordia (default), domain 10", domain = 10,
       metric_method = "direct_extended", color = "#d62728")
)

reference_label = "Concordia (default), domain 10"

other_color = c("Sensitivity settings" = "grey30", "Number of domains" = "#2c7fb8")

for (cur_csv in c(fraction_csv, metric_csv, match_metric_csv)){
  if (!file.exists(cur_csv)){
    stop("missing ", cur_csv, "; run the script that writes it first")
  }
}

df_fraction = fread(fraction_csv, select = c("region_id", "mean_prop_t"))
stopifnot(!any(duplicated(df_fraction$region_id)))

high_ids = df_fraction[!is.na(mean_prop_t) & mean_prop_t >= min_mean_prop_t, region_id]

cat("images at or above ", format(min_mean_prop_t, nsmall = 2), " on mean prop_t: ",
    length(high_ids), "\n", sep = "")

df_metric_all = fread(metric_csv,
                      select = c("method", "region_id", "domain",
                                 "proportion_below_threshold"))

df_match_metric = fread(match_metric_csv,
                        select = c("group", "setting", "setting_label", "criterion",
                                   "region_id", "domain", "proportion_below_threshold"))

stopifnot(all(df_match_metric[setting == "permutate", setting_label] ==
                "permuted data, permutate"))
df_match_metric[setting == "permutate", setting_label := "permuted data, permutate10pct"]

stopifnot(all(df_match_metric$region_id %in% df_metric_all$region_id))

box_rows = list()

for (target in method_targets){

  stopifnot(target$metric_method %in% df_metric_all$method)

  df_cur = df_metric_all[method == target$metric_method & domain == target$domain &
                           !is.na(proportion_below_threshold) & region_id %in% high_ids]

  stopifnot(!any(duplicated(df_cur$region_id)))

  box_rows[[length(box_rows) + 1]] = data.table(
    group = "Methods", label = target$label, color = target$color,
    region_id = df_cur$region_id, f_lin = df_cur$proportion_below_threshold)
}

pair_list = unique(df_match_metric[, .(group, setting, setting_label, criterion, domain)])
pair_list[, setting_order := match(setting, unique(setting))]
setorder(pair_list, setting_order, domain)

for (pair_index in seq_len(nrow(pair_list))){

  cur_pair = pair_list[pair_index]

  cur_group = if (cur_pair$group == "sensitivity") "Sensitivity settings" else "Number of domains"

  cur_label_head = if (cur_group == "Sensitivity settings"){
    paste0(toupper(substr(cur_pair$setting_label, 1, 1)),
           substr(cur_pair$setting_label, 2, nchar(cur_pair$setting_label)))
  }else{
    cur_pair$setting_label
  }

  cur_label = paste0(cur_label_head, ", domain ", cur_pair$domain,
                     if (cur_pair$criterion == "column") " (column)" else "")

  df_cur = df_match_metric[setting == cur_pair$setting & domain == cur_pair$domain &
                             !is.na(proportion_below_threshold) & region_id %in% high_ids]

  stopifnot(!any(duplicated(df_cur$region_id)))

  box_rows[[length(box_rows) + 1]] = data.table(
    group = cur_group, label = cur_label, color = other_color[[cur_group]],
    region_id = df_cur$region_id, f_lin = df_cur$proportion_below_threshold)
}

df_box = rbindlist(box_rows)

label_order = unique(df_box$label)

stopifnot(!any(duplicated(label_order)))
stopifnot(reference_label %in% label_order)

df_box[, label := factor(label, levels = rev(label_order))]
df_box[, group := factor(group, levels = group_levels)]

stopifnot(!any(is.na(df_box$f_lin)), !any(is.na(df_box$group)))

df_count = df_box[, .(n_images = .N, median_f_lin = median(f_lin)), by = .(group, label)]

fwrite(df_box[, .(group, label, region_id, f_lin)],
       file.path(figure_dir, "f_lin_by_image.csv"))

reference_median = df_count[label == reference_label, median_f_lin]

options(width = 200)
cat("\nimages and median f_lin per box; reference median ",
    format(round(reference_median, 3), nsmall = 3), "\n", sep = "")
print(df_count[order(match(as.character(label), label_order)),
               .(group, label, n_images, median_f_lin = round(median_f_lin, 3))],
      row.names = FALSE)

color_of = setNames(unique(df_box[, .(label, color)])$color,
                    unique(df_box[, .(label, color)])$label)

p_box = ggplot(df_box, aes(x = f_lin, y = label, color = label)) +
  geom_vline(xintercept = reference_median, linetype = "dashed",
             colour = "#d62728", linewidth = 0.4) +
  geom_boxplot(outlier.shape = NA, fill = NA, width = 0.6, linewidth = 0.5) +
  geom_point(position = position_jitter(height = 0.15, width = 0, seed = 1),
             size = 0.9, alpha = 0.45) +
  geom_text(data = df_count, aes(x = count_x, y = label, label = paste0("n = ", n_images)),
            inherit.aes = FALSE, hjust = 0, size = count_font_size / .pt,
            colour = "grey25") +
  scale_color_manual(values = color_of, guide = "none") +
  scale_x_continuous(breaks = seq(0, 1, by = 0.25)) +
  coord_cartesian(xlim = c(0, x_upper)) +
  facet_grid(group ~ ., scales = "free_y", space = "free_y") +
  labs(x = expression(italic(f)[lin]), y = NULL) +
  theme_classic(base_size = base_font_size) +
  theme(strip.text.y = element_text(size = strip_font_size, angle = -90),
        strip.background = element_rect(colour = NA, fill = "grey92"),
        panel.spacing = unit(8, "pt"),
        plot.margin = margin(5.5, 5.5, 5.5, 5.5))

figure_file = file.path(figure_dir, "f_lin_boxplots.pdf")

while (!is.null(dev.list())) dev.off()

pdf(figure_file, width = figure_width, height = figure_height)
print(p_box)
invisible(dev.off())

cat("\nsaved ", figure_file, "\n", sep = "")
cat("saved ", file.path(figure_dir, "f_lin_by_image.csv"), "\n", sep = "")
