library(data.table)

output_dir = file.path("../revision_figures",
                       "supp_default_domain10_matches_across_settings")
if (!dir.exists(output_dir)){
  dir.create(output_dir, recursive = TRUE)
}

target_domain = "10"
threshold = 0.50

sensitivity_dir = "../revision_figures/supp_crosstable_cords_concordia_vs_sensitivity_settings"
n_domains_dir = "../revision_figures/supp_crosstable_cords_concordia_vs_n_domains"


setting_list = list(
  list(group = "sensitivity", label = "distance cutoff 14.5, radius 43.5", tag = "dist_14p5"),
  list(group = "sensitivity", label = "distance cutoff 17, radius 51", tag = "dist_17"),
  list(group = "sensitivity", label = "extension radius 32", tag = "rad_32"),
  list(group = "sensitivity", label = "extension radius 64", tag = "rad_64"),
  list(group = "sensitivity", label = "top k 3", tag = "topk_3"),
  list(group = "sensitivity", label = "top k 5", tag = "topk_5"),
  list(group = "sensitivity", label = "composition cutoff 0.157", tag = "ctg_0p157"),
  list(group = "sensitivity", label = "composition cutoff 0.202", tag = "ctg_0p202"),
  list(group = "sensitivity", label = "degree limit 16", tag = "deg_16"),
  list(group = "sensitivity", label = "degree limit 24", tag = "deg_24"),
  list(group = "sensitivity", label = "cell type groups group8", tag = "group8"),
  list(group = "sensitivity", label = "permuted data, permutate5pct", tag = "permutate5pct"),
  list(group = "sensitivity", label = "permuted data, permutate", tag = "permutate"),

  list(group = "n_domains", label = "k = 8", tag = "k8"),
  list(group = "n_domains", label = "k = 12", tag = "k12"),
  list(group = "n_domains", label = "k = 16", tag = "k16"),
  list(group = "n_domains", label = "k = 20", tag = "k20")
)

stopifnot(!any(duplicated(sapply(setting_list, function(x) x$tag))))

read_crosstable <- function(cur_dir, cur_prefix, cur_tag){

  cur_csv = file.path(cur_dir, paste0(cur_prefix, cur_tag, ".csv"))

  if (!file.exists(cur_csv)){
    stop("missing ", cur_csv, "; run the crosstable script that writes it first")
  }

  as.matrix(read.csv(cur_csv, row.names = 1, check.names = FALSE))
}


match_rows = list()

for (setting in setting_list){

  cur_dir = if (setting$group == "sensitivity") sensitivity_dir else n_domains_dir

  counts = read_crosstable(cur_dir, "counts_", setting$tag)
  by_row = read_crosstable(cur_dir, "by_row_sum_", setting$tag)
  by_col = read_crosstable(cur_dir, "by_column_sum_", setting$tag)

  stopifnot(identical(dim(counts), dim(by_row)), identical(dim(counts), dim(by_col)))
  stopifnot(identical(dimnames(counts), dimnames(by_row)),
            identical(dimnames(counts), dimnames(by_col)))
  stopifnot(target_domain %in% rownames(counts))
  stopifnot(isTRUE(all.equal(by_row, sweep(counts, 1, rowSums(counts), "/"),
                             tolerance = 1e-9, check.attributes = FALSE)))
  stopifnot(isTRUE(all.equal(by_col, sweep(counts, 2, colSums(counts), "/"),
                             tolerance = 1e-9, check.attributes = FALSE)))

  row_share = by_row[target_domain, ]
  col_share = by_col[target_domain, ]

  is_match = row_share > threshold | col_share > threshold

  if (!any(is_match)){

    match_rows[[length(match_rows) + 1]] = data.table(
      group = setting$group, setting = setting$label, tag = setting$tag,
      n_domains = ncol(counts), domain = NA_character_,
      n_cells_shared = NA_integer_, n_cells_domain = NA_integer_,
      share_of_default_domain10 = NA_real_, share_of_domain_in_default10 = NA_real_,
      criterion = "no match")

  }else{

    for (cur_domain in colnames(counts)[is_match]){

      meets_row = row_share[[cur_domain]] > threshold
      meets_col = col_share[[cur_domain]] > threshold

      match_rows[[length(match_rows) + 1]] = data.table(
        group = setting$group, setting = setting$label, tag = setting$tag,
        n_domains = ncol(counts), domain = cur_domain,
        n_cells_shared = as.integer(counts[target_domain, cur_domain]),
        n_cells_domain = as.integer(sum(counts[, cur_domain])),
        share_of_default_domain10 = row_share[[cur_domain]],
        share_of_domain_in_default10 = col_share[[cur_domain]],
        criterion = if (meets_row && meets_col) "both" else if (meets_row) "row" else "column")
    }
  }
}

df_match = rbindlist(match_rows)

output_csv = file.path(output_dir, "default_domain10_matches.csv")

write.csv(df_match, output_csv, row.names = FALSE)


cat("default domain 10 holds ",
    format(sum(read_crosstable(sensitivity_dir, "counts_", "dist_14p5")[target_domain, ]),
           big.mark = ","),
    " cells; a match has more than ", threshold, " in row 10 of either table\n\n", sep = "")

print(df_match[, .(setting, domain,
                   share_of_default10 = round(share_of_default_domain10, 3),
                   share_in_default10 = round(share_of_domain_in_default10, 3),
                   criterion)],
      row.names = FALSE)

cat("\nsaved", output_csv, "\n")
