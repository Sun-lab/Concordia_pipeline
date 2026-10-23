library(data.table)
library(dplyr)
library(survival)
library(ggplot2)

theme_set(theme_classic())
options(bitmapType = "cairo")

figure_dir = file.path("../revision_figures",
                       "figure3_domain_share_OS_and_cox_zph_three_methods_strata_age65_cell_weighted")
if (!dir.exists(figure_dir)){
  dir.create(figure_dir, recursive = TRUE)
}

analysis_dir = "../revision/analysis/figures"

figure_width = 2.5
figure_height = 2.2
base_font_size = 10
pvalue_font_size = 8.5

age_cut = 65
significance_level = 0.05

setting_list = list(
  list(label = "Concordia-Basic",
       share_csv = "domain_share_per_image_basic3_vs_basic10.csv",
       column = "share_domain_3_tumor",
       color = "#0072B2"),
  list(label = "Concordia",
       share_csv = "domain_share_per_image_direct8_vs_direct3.csv",
       column = "share_domain_8_tumor",
       color = "#CC79A7"),
  list(label = "K-means",
       share_csv = "domain_share_per_image_kmeans3_vs_kmeans7.csv",
       column = "share_domain_3_tumor",
       color = "#E69F00")
)

setting_levels = sapply(setting_list, function(x) x$label)
setting_colors = setNames(sapply(setting_list, function(x) x$color), setting_levels)

stopifnot(!any(duplicated(setting_levels)))

safe_zph <- function(cur_fit){
  tryCatch(cox.zph(cur_fit)$table,
           error = function(e) NULL,
           warning = function(w) suppressWarnings(cox.zph(cur_fit)$table))
}

age_group_of <- function(cur_age){
  factor(ifelse(cur_age >= age_cut, "at_or_above", "below"),
         levels = c("below", "at_or_above"))
}


df_images = read.csv("../data/Cords_data/patient_image_greq_1000_cells.csv", header = TRUE)
df_images$RoiID = gsub(",", "_", df_images$RoiID)
df_images = df_images[which(df_images$DX.name %in% c("Adenocarcinoma", "Squamous cell carcinoma")), ]

df_patients = read.csv("../data/Cords_data/patient_features.csv", header = TRUE)
df_patients_subset = df_patients[which(df_patients$Patient_ID %in% df_images$Patient_ID), ]

df_patients_matched = df_patients_subset[match(df_images$Patient_ID, df_patients_subset$Patient_ID), ]
stopifnot(all(df_images$Patient_ID == df_patients_matched$Patient_ID))

df_roi = cbind(df_patients_matched, df_images[, c("RoiID", "n_cells")])

cat("clinical table: ", nrow(df_roi), " images over ",
    length(unique(df_roi$Patient_ID)), " patients\n", sep = "")


result_rows = NULL

for (setting in setting_list){

  share_path = file.path(analysis_dir, setting$share_csv)

  if (!file.exists(share_path)){
    stop("missing ", share_path)
  }

  df_share = fread(share_path)

  stopifnot(setting$column %in% names(df_share))
  stopifnot(!any(duplicated(df_share$region_id)))

  df_merged = merge(df_share[, c("region_id", setting$column), with = FALSE],
                    df_roi, by.x = "region_id", by.y = "RoiID")

  df_merged = df_merged[!is.na(get(setting$column))]
  df_merged[, metric_value := get(setting$column)]

  df_patient = as.data.frame(df_merged) %>%
    group_by(Patient_ID) %>%
    summarise(metric_value = weighted.mean(metric_value, w = n_cells),
              n_images_used = n(),
              across(c("Ev.O", "OS", "DX.name", "Age", "Gender", "Grade", "Smok"),
                     unique),
              .groups = "drop") %>%
    as.data.frame()

  stopifnot(!any(is.na(df_patient$metric_value)))
  stopifnot(all(df_merged$n_cells > 0))

  df_prop = data.frame(prop = df_patient$metric_value,
                       Ev.O = df_patient$Ev.O,
                       OS = df_patient$OS,
                       DX.name = df_patient$DX.name,
                       Gender = factor(df_patient$Gender, levels = c(1, 2)),
                       age_group = age_group_of(df_patient$Age),
                       Grade = factor(df_patient$Grade, levels = c(1, 2, 3)),
                       Smok = factor(df_patient$Smok, levels = c(0, 1, 2, 3)))

  stopifnot(sum(df_prop$age_group == "below", na.rm = TRUE) > 0,
            sum(df_prop$age_group == "at_or_above", na.rm = TRUE) > 0)

  cur_fit = coxph(Surv(OS, Ev.O) ~ prop + strata(age_group) + Gender +
                    DX.name + Grade + Smok,
                  data = df_prop)

  cur_summary = summary(cur_fit)

  cur_zph = safe_zph(cur_fit)

  stopifnot(is.null(cur_zph) || !any(grepl("age", rownames(cur_zph), ignore.case = TRUE)))

  cur_row = data.frame(
    setting = setting$label,
    metric = setting$column,
    n_images = nrow(df_merged),
    n_patients = nrow(df_patient),
    n_patients_two_images = sum(df_patient$n_images_used >= 2),
    n_patients_used = cur_fit$n,
    n_events_used = cur_fit$nevent,
    hazard_ratio = cur_summary$coefficients["prop", "exp(coef)"],
    hazard_ratio_lower95 = cur_summary$conf.int["prop", "lower .95"],
    hazard_ratio_upper95 = cur_summary$conf.int["prop", "upper .95"],
    pvalue = cur_summary$coefficients["prop", "Pr(>|z|)"],
    zph_global_pvalue = if (is.null(cur_zph)) NA_real_ else cur_zph["GLOBAL", "p"])

  result_rows = rbind(result_rows, cur_row)
}


result_rows$setting = factor(result_rows$setting, levels = setting_levels)
result_rows$logp = -log10(result_rows$pvalue)
result_rows$p_label = paste0("p=", formatC(signif(result_rows$pvalue, 2), format = "g"))
result_rows$zph_logp = -log10(result_rows$zph_global_pvalue)
result_rows$zph_p_label = paste0("p=", formatC(signif(result_rows$zph_global_pvalue, 2), format = "g"))

write.csv(result_rows[, setdiff(names(result_rows), c("logp", "p_label", "zph_logp", "zph_p_label"))],
          file.path(figure_dir, "pvalues.csv"), row.names = FALSE)

options(width = 200)
print(result_rows[, c("setting", "n_images", "n_patients_used", "n_events_used",
                      "hazard_ratio", "hazard_ratio_lower95", "hazard_ratio_upper95",
                      "pvalue", "zph_global_pvalue")],
      row.names = FALSE, digits = 3)

build_barchart <- function(df_plot, y_column, label_column){

  y_upper = max(c(df_plot[[y_column]], -log10(significance_level)), na.rm = TRUE) * 1.18

  ggplot(df_plot[!is.na(df_plot[[y_column]]), ],
         aes(x = setting, y = .data[[y_column]], fill = setting)) +
    geom_col(width = 0.65) +
    geom_hline(yintercept = -log10(significance_level), linetype = "dashed",
               colour = "grey30", linewidth = 0.3) +
    geom_text(aes(label = .data[[label_column]]), vjust = -0.4,
              size = pvalue_font_size / .pt) +
    scale_fill_manual(values = setting_colors, guide = "none") +
    scale_x_discrete(drop = FALSE) +
    scale_y_continuous(expand = expansion(mult = c(0, 0)), limits = c(0, y_upper),
                       breaks = seq(0, ceiling(y_upper * 2) / 2, by = 0.5),
                       labels = function(x) sprintf("%.1f", x)) +
    labs(x = NULL, y = expression(-log[10] * "(p)")) +
    theme_classic(base_size = base_font_size, base_line_size = 0.3) +
    theme(axis.text.x = element_text(colour = "black", angle = 30, hjust = 1),
          axis.text.y = element_text(colour = "black"),
          axis.ticks = element_line(colour = "black"),
          plot.margin = margin(4, 4, 2, 4))
}

figure_list = list(
  "pvalue_barchart.pdf" = build_barchart(result_rows, "logp", "p_label"),
  "cox_zph_global_pvalue_barchart.pdf" = build_barchart(result_rows, "zph_logp", "zph_p_label"))

while (!is.null(dev.list())) dev.off()

for (cur_name in names(figure_list)){
  figure_file = file.path(figure_dir, cur_name)
  pdf(figure_file, width = figure_width, height = figure_height)
  print(figure_list[[cur_name]])
  invisible(dev.off())
  cat("saved ", figure_file, "\n", sep = "")
}
cat("saved ", file.path(figure_dir, "pvalues.csv"), "\n", sep = "")
