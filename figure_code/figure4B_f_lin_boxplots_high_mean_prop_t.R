library(data.table)
library(ggplot2)
library(grid)

theme_set(theme_classic())
options(bitmapType = "cairo")

figure_dir = file.path("../revision_figures",
                       "figure5_f_lin_boxplots_high_mean_prop_t")
if (!dir.exists(figure_dir)){
  dir.create(figure_dir, recursive = TRUE)
}

n_domains = 10

min_mean_prop_t = 0.85

figure_width = 6.4
figure_height = 4.8

base_font_size = 20
legend_font_size = 17
title_font_size = 20

pvalue_font_size = 14

axis_line_top = 1.02

metric_dir = "../revision/analysis/figures"

fraction_csv = file.path(metric_dir,
                         "tumor_neighbor_tumor_fraction_sampled_images_random_sample.csv")
metric_csv = file.path(metric_dir,
                       "local_pca_dimension_cell_min_all_methods_random_sample.csv")

targets = list(
  list(tag = "raw_kmeans", short = "K-means",
       domain = 6, metric_method = "raw_kmeans"),
  list(tag = "basic", short = "Concordia-Basic",
       domain = 7, metric_method = "direct_basic"),
  list(tag = "local", short = "Concordia-Local",
       domain = 6, metric_method = "direct_local"),
  list(tag = "extended", short = "Concordia",
       domain = 10, metric_method = "direct_extended")
)

reference_target = targets[[length(targets)]]

stopifnot(!any(duplicated(sapply(targets, function(x) x$tag))))

method_colors = c("#1f77b4", "#9467bd", "#8c564b", "#d62728")

stopifnot(length(method_colors) == length(targets))



if (!file.exists(fraction_csv)){
  stop("missing ", fraction_csv,
       "; run tumor_neighbor_tumor_fraction_sampled_images_random_sample.py first")
}

if (!file.exists(metric_csv)){
  stop("missing ", metric_csv,
       "; run local_pca_dimension_sampled_images_all_methods_cell_min_random_sample.py first")
}

df_fraction = fread(fraction_csv, select = c("region_id", "mean_prop_t"))

stopifnot(!any(duplicated(df_fraction$region_id)))

df_fraction = df_fraction[!is.na(mean_prop_t)]

df_metric_all = fread(metric_csv,
                      select = c("method", "region_id", "domain",
                                 "proportion_below_threshold"))

stopifnot(all(df_metric_all$domain %in% 1:n_domains))

df_wide = df_fraction

for (target in targets){

  stopifnot(target$metric_method %in% df_metric_all$method)

  df_cur = df_metric_all[method == target$metric_method &
                           domain == target$domain &
                           !is.na(proportion_below_threshold),
                         .(region_id, f_lin = proportion_below_threshold)]

  stopifnot(!any(duplicated(df_cur$region_id)))

  setnames(df_cur, "f_lin", paste0("f_lin_", target$tag))

  df_wide = merge(df_wide, df_cur, by = "region_id")
}

cat("images with every annotation measured: ", nrow(df_wide), "\n", sep = "")

df_wide = df_wide[mean_prop_t >= min_mean_prop_t]

cat("of those, at or above ", format(min_mean_prop_t, nsmall = 2),
    " on mean prop_t: ", nrow(df_wide), "\n", sep = "")

if (nrow(df_wide) < 3){
  stop("only ", nrow(df_wide), " images clear the cut, too few to summarise")
}



method_labels = sapply(targets, function(x) x$short)

names(method_colors) = method_labels

df_box = rbindlist(lapply(targets, function(target){
  data.table(region_id = df_wide$region_id,
             method = target$short,
             f_lin = df_wide[[paste0("f_lin_", target$tag)]])
}))

df_box[, method := factor(method, levels = method_labels)]

stopifnot(!any(is.na(df_box$f_lin)))

fwrite(df_box, file.path(figure_dir, "f_lin_by_image.csv"))



reference_values = df_wide[[paste0("f_lin_", reference_target$tag)]]

cat("\n==== ", nrow(df_wide), " images, medians and paired comparisons ====\n",
    sep = "")

for (target in targets){
  cat("median ", target$short, ": ",
      format(round(median(df_wide[[paste0("f_lin_", target$tag)]]), 3), nsmall = 3),
      "\n", sep = "")
}

paired_of = list()

for (target in targets[-length(targets)]){

  cur_values = df_wide[[paste0("f_lin_", target$tag)]]
  cur_difference = reference_values - cur_values

  cur_test = suppressWarnings(wilcox.test(reference_values, cur_values,
                                          paired = TRUE))

  paired_of[[target$tag]] = list(median_difference = median(cur_difference),
                                 n_higher = sum(cur_difference > 0),
                                 p_value = cur_test$p.value)

  cat(reference_target$short, " - ", target$short,
      ": median per-image difference ",
      format(round(median(cur_difference), 3), nsmall = 3),
      ", higher in ", reference_target$short, " for ",
      sum(cur_difference > 0), " of ", length(cur_difference),
      " images, paired signed rank p = ",
      format(signif(cur_test$p.value, 3)), "\n", sep = "")
}



bracket_base = 1.04
bracket_step = 0.11
bracket_tick = 0.025

x_of = setNames(seq_along(method_labels), method_labels)

x_reference = x_of[[reference_target$short]]

member_targets = targets[-length(targets)]
member_targets = member_targets[order(sapply(member_targets, function(x){
  abs(x_of[[x$short]] - x_reference)
}))]

bracket_segments = NULL
bracket_labels = NULL

for (member_index in seq_along(member_targets)){

  target = member_targets[[member_index]]

  x_other = x_of[[target$short]]
  y_bracket = bracket_base + (member_index - 1) * bracket_step

  cur_paired = paired_of[[target$tag]]

  bracket_segments = rbind(bracket_segments, data.frame(
    x = c(min(x_other, x_reference), min(x_other, x_reference), max(x_other, x_reference)),
    xend = c(max(x_other, x_reference), min(x_other, x_reference), max(x_other, x_reference)),
    y = c(y_bracket, y_bracket, y_bracket),
    yend = c(y_bracket, y_bracket - bracket_tick, y_bracket - bracket_tick)))

  bracket_labels = rbind(bracket_labels, data.frame(
    x = (x_other + x_reference) / 2,
    y = y_bracket + 0.012,
    label = paste0("p = ", format(signif(cur_paired$p_value, 2)))))
}

bracket_top = bracket_base + (length(member_targets) - 1) * bracket_step + 0.07



p_box = ggplot(df_box, aes(x = method, y = f_lin, color = method)) +
  geom_boxplot(outlier.shape = NA, fill = NA, width = 0.6, linewidth = 0.6) +
  geom_point(position = position_jitter(width = 0.12, seed = 1),
             size = 1.4, alpha = 0.55) +
  scale_color_manual(values = method_colors, drop = FALSE) +
  geom_segment(data = bracket_segments,
               aes(x = x, xend = xend, y = y, yend = yend),
               inherit.aes = FALSE, colour = "grey30", linewidth = 0.4) +
  geom_text(data = bracket_labels,
            aes(x = x, y = y, label = label),
            inherit.aes = FALSE, colour = "grey20",
            size = pvalue_font_size / .pt, vjust = 0) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.25)) +
  coord_cartesian(ylim = c(0, bracket_top)) +
  labs(x = NULL, y = NULL, title = expression(italic(f)[lin]), color = NULL) +
  annotate("segment", x = -Inf, xend = -Inf, y = 0, yend = axis_line_top,
           colour = "black", linewidth = 0.5) +
  theme_classic(base_size = base_font_size) +
  theme(axis.line.y = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        plot.title = element_text(size = title_font_size, hjust = 0.5),
        legend.position = "right",
        legend.text = element_text(size = legend_font_size),
        legend.key.height = unit(1.3, "lines"))

figure_file = file.path(figure_dir, "f_lin_boxplots.pdf")

while (!is.null(dev.list())) dev.off()

pdf(figure_file, width = figure_width, height = figure_height)
print(p_box)
invisible(dev.off())

cat("\nsaved ", figure_file, "\n", sep = "")
cat("saved ", file.path(figure_dir, "f_lin_by_image.csv"), "\n", sep = "")
