library(data.table)
library(ggplot2)

theme_set(theme_classic())
options(bitmapType = "cairo")

figure_dir = file.path("../revision_figures",
                       "supp_domain_coord_dist_heatmap_direct_domains_danenberg")
if (!dir.exists(figure_dir)){
  dir.create(figure_dir, recursive = TRUE)
}

figure_width = 5.2
figure_height = 4.9
base_font_size = 12

value_font_size = 3
legend_font_size = 9

n_domains = 10

dist_csv = paste0("../revision/code/results_rs_ensemble_scaled_direct_domains/",
                  "danenberg_2022__group_default__base20__rad60__topk4__ctg0p176__deg20",
                  "__ensemble_10/extended/domains_k10/summary_coord_dist_matrix.csv")

na_fill = "grey90"


if (!file.exists(dist_csv)){
  stop("missing ", dist_csv)
}

dist_mat = as.matrix(read.csv(dist_csv, header = TRUE, row.names = 1,
                              check.names = FALSE))

domain_names = paste0("domain_", 1:n_domains)

stopifnot(all(dim(dist_mat) == c(n_domains, n_domains)))
stopifnot(identical(rownames(dist_mat), domain_names))
stopifnot(identical(colnames(dist_mat), domain_names))
stopifnot(isTRUE(all.equal(dist_mat, t(dist_mat))))
stopifnot(all(is.na(diag(dist_mat))))

n_missing = sum(is.na(dist_mat[upper.tri(dist_mat)]))
cat("pairs with a median: ", n_domains*(n_domains-1)/2 - n_missing,
    "; NA pairs: ", n_missing, "\n", sep = "")
cat("range of off-diagonal distances: ",
    paste(round(range(dist_mat, na.rm = TRUE), 2), collapse = " to "), "\n", sep = "")


combined = data.table(
  domain_i = rep(1:n_domains, times = n_domains),
  domain_j = rep(1:n_domains, each = n_domains),
  dist = c(dist_mat))

combined[, domain_i := factor(domain_i, levels = rev(1:n_domains))]
combined[, domain_j := factor(domain_j, levels = 1:n_domains)]

dist_mid = mean(range(combined$dist, na.rm = TRUE))
combined[, label := ifelse(is.na(dist), "", sprintf("%.0f", dist))]
combined[, label_color := ifelse(!is.na(dist) & dist < dist_mid, "black", "white")]

p1 = ggplot(combined, aes(x = domain_j, y = domain_i, fill = dist)) +
  geom_tile(color = "white", linewidth = 0.4) +
  geom_text(aes(label = label, color = label_color), size = value_font_size) +
  scale_color_identity() +
  scale_fill_viridis_c(option = "viridis", direction = -1, na.value = na_fill,
                       name = NULL) +
  guides(fill = guide_colorbar(barwidth = unit(0.35, "cm"),
                               barheight = unit(3.5, "cm"))) +
  scale_x_discrete(expand = expansion(add = 0), position = "top") +
  scale_y_discrete(expand = expansion(add = 0)) +
  coord_fixed() +
  labs(title = "Breast cancer dataset\nConcordia domains, median physical distance",
       x = "domain", y = "domain") +
  theme_minimal(base_size = base_font_size) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        panel.grid = element_blank(),
        legend.title = element_text(size = legend_font_size),
        legend.text = element_text(size = legend_font_size),
        plot.margin = margin(5.5, 5.5, 5.5, 5.5))

figure_file = file.path(figure_dir, "heatmap.pdf")

while (!is.null(dev.list())) dev.off()

pdf(figure_file, width = figure_width, height = figure_height)
print(p1)
dev.off()

table_file = file.path(figure_dir, "domain_coord_dist_median.csv")
write.csv(combined[, .(domain_i, domain_j, dist)], table_file, row.names = FALSE)

cat("\nsaved", figure_file, "\n")
cat("saved", table_file, "\n")
