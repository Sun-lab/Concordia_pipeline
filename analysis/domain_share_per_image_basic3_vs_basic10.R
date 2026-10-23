library(data.table)

figure_dir = "figures"
if (!dir.exists(figure_dir)){
  dir.create(figure_dir, recursive = TRUE)
}

output_csv = file.path(figure_dir, "domain_share_per_image_basic3_vs_basic10.csv")

raw_dir = "../../data/Cords_data/raw_data"
data_dir = "../../data/Cords_data"
region_list_csv = file.path(data_dir, "region_list.csv")

direct_root = "../code/results_rs_ensemble_scaled_direct_domains/"

n_domains = 10
n_repeats = 10

default_run_id = "cords_2024__group_default__base16__rad48__topk4__ctg0p176__deg20"

annotation_csv = paste0(direct_root, default_run_id, "__ensemble_", n_repeats,
                        "/basic/domains_k", n_domains,
                        "/domain_annotation_for_cells.csv")

count_csv = paste0(direct_root, default_run_id, "__ensemble_", n_repeats,
                   "/basic/domains_k", n_domains,
                   "/cluster_hard_count_domains.csv")

annotation_label = "direct k = 10, basic"

domain_a = 3
domain_b = 10

target_ct_group = "tumor"

min_domain_cells = 30

example_floors = c(100, 300)

stopifnot(domain_a != domain_b)
stopifnot(domain_a %in% 1:n_domains, domain_b %in% 1:n_domains)


source("../code/data_utilities.R")

feature_list = data_features(data_name = "cords_2024", data_dir = data_dir)

group_ct_mapping = feature_list$group_ct_mapping

all_ct_groups = names(group_ct_mapping)

if (!(target_ct_group %in% all_ct_groups)){
  stop("no cell type group named ", target_ct_group, "; the groups are ",
       paste(sort(all_ct_groups), collapse = ", "))
}

ct_group_of = unlist(lapply(all_ct_groups, function(ct_grp){
  setNames(rep(ct_grp, length(group_ct_mapping[[ct_grp]])), group_ct_mapping[[ct_grp]])
}))

stopifnot(!any(duplicated(names(ct_group_of))))

stopifnot(normalizePath(raw_dir) == normalizePath(feature_list$raw_dir))

cat("the ", target_ct_group, " group holds ",
    length(group_ct_mapping[[target_ct_group]]), " cell types: ",
    paste(sort(group_ct_mapping[[target_ct_group]]), collapse = ", "), "\n", sep = "")


region_ids = read.csv(region_list_csv, header = TRUE)$region_ID

stopifnot(!any(duplicated(region_ids)))

cat("region list: ", length(region_ids), " images\n", sep = "")

cell_table = rbindlist(lapply(region_ids, function(cur_region_id){
  df_cur = fread(file.path(raw_dir, paste0(cur_region_id, ".csv")),
                 select = c("CELL_ID", "CELL_TYPE"))
  df_cur[, region_id := cur_region_id]
  df_cur
}))

stopifnot(!any(duplicated(cell_table$CELL_ID)))

cell_table[, ct_group := ct_group_of[CELL_TYPE]]
stopifnot(!any(is.na(cell_table$ct_group)))

cell_table[, is_target_ct := ct_group == target_ct_group]

cat("read ", nrow(cell_table), " cells over ",
    length(unique(cell_table$region_id)), " images, ",
    sum(cell_table$is_target_ct), " of them ", target_ct_group, "\n", sep = "")


if (!file.exists(annotation_csv)){
  stop("missing ", annotation_csv)
}

df_ann = fread(annotation_csv, select = c("CELL_ID", "domain"))

df_ann[, domain := as.integer(domain)]

stopifnot(all(df_ann$domain %in% 1:n_domains))
stopifnot(!any(duplicated(df_ann$CELL_ID)))

setkey(df_ann, CELL_ID)

cell_table[, domain := df_ann[.(cell_table$CELL_ID), domain]]

stopifnot(!any(is.na(cell_table$domain)))

cat("read ", nrow(df_ann), " annotated cells for ", annotation_label,
    ", domains ", min(df_ann$domain), " to ", max(df_ann$domain), "\n", sep = "")


df_result = cell_table[, .(
  n_cells_image = .N,
  n_a       = sum(domain == domain_a),
  n_b       = sum(domain == domain_b),
  n_a_target = sum(domain == domain_a & is_target_ct),
  n_b_target = sum(domain == domain_b & is_target_ct)),
  by = region_id]

stopifnot(all(df_result$n_a_target <= df_result$n_a))
stopifnot(all(df_result$n_b_target <= df_result$n_b))

df_result[, n_pair := n_a + n_b]
df_result[, n_pair_target := n_a_target + n_b_target]

df_result[, ok_all := n_a >= min_domain_cells & n_b >= min_domain_cells]
df_result[, ok_target := n_a_target >= min_domain_cells &
            n_b_target >= min_domain_cells]

df_result[, share_all := ifelse(ok_all, n_a / n_pair, NA_real_)]
df_result[, share_target := ifelse(ok_target, n_a_target / n_pair_target, NA_real_)]

reason_of <- function(n_first, n_second, what){
  fifelse(n_first < min_domain_cells & n_second < min_domain_cells,
          paste0("fewer than ", min_domain_cells, " ", what, " in both domains"),
          fifelse(n_first < min_domain_cells,
                  paste0("fewer than ", min_domain_cells, " ", what,
                         " in domain ", domain_a),
                  fifelse(n_second < min_domain_cells,
                          paste0("fewer than ", min_domain_cells, " ", what,
                                 " in domain ", domain_b),
                          "")))
}

df_result[, not_measured_all := reason_of(n_a, n_b, "cells")]
df_result[, not_measured_target := reason_of(n_a_target, n_b_target,
                                             paste(target_ct_group, "cells"))]

stopifnot(identical(is.na(df_result$share_all), df_result$not_measured_all != ""))
stopifnot(identical(is.na(df_result$share_target),
                    df_result$not_measured_target != ""))

stopifnot(!any(is.na(df_result$share_all) & !is.na(df_result$share_target)))

df_result[, c("ok_all", "ok_target") := NULL]

setnames(df_result,
         c("n_a", "n_b", "n_a_target", "n_b_target", "share_all", "share_target"),
         c(paste0("n_domain_", domain_a),
           paste0("n_domain_", domain_b),
           paste0("n_domain_", domain_a, "_", target_ct_group),
           paste0("n_domain_", domain_b, "_", target_ct_group),
           paste0("share_domain_", domain_a, "_all_cells"),
           paste0("share_domain_", domain_a, "_", target_ct_group)))

setorder(df_result, region_id)


if (file.exists(count_csv)){

  image_count = fread(count_csv)
  setnames(image_count, 1, "region_id")

  for (cur_domain in c(domain_a, domain_b)){

    cur_column = paste0("domain_", cur_domain)
    stopifnot(cur_column %in% names(image_count))

    df_compare = merge(df_result[, .(region_id,
                                     here = get(paste0("n_domain_", cur_domain)))],
                       image_count[, .(region_id, there = get(cur_column))],
                       by = "region_id")

    if (nrow(df_compare) != nrow(df_result)){
      stop(count_csv, " does not cover every image counted here")
    }

    if (!identical(as.integer(df_compare$here), as.integer(df_compare$there))){
      stop("the counts for domain ", cur_domain, " disagree with ", count_csv,
           " on ", sum(df_compare$here != df_compare$there), " images")
    }
  }

  cat("counts for domains ", domain_a, " and ", domain_b, " agree with ",
      count_csv, " on all ", nrow(df_result), " images\n", sep = "")

}else{
  cat("note: ", count_csv, " is not there, so the counts were not cross checked\n",
      sep = "")
}


fwrite(df_result, output_csv)


share_all_column = paste0("share_domain_", domain_a, "_all_cells")
share_target_column = paste0("share_domain_", domain_a, "_", target_ct_group)

cat("\n==== ", nrow(df_result), " images ====\n", sep = "")

for (cur_column in c(share_all_column, share_target_column)){

  cur_values = df_result[[cur_column]]
  cur_values = cur_values[!is.na(cur_values)]

  cat("\n", cur_column, ": ", length(cur_values), " images measured, ",
      nrow(df_result) - length(cur_values), " without a denominator\n", sep = "")
  cat("  mean ", format(round(mean(cur_values), 3), nsmall = 3),
      ", median ", format(round(median(cur_values), 3), nsmall = 3),
      ", quartiles ", format(round(quantile(cur_values, 0.25), 3), nsmall = 3),
      " and ", format(round(quantile(cur_values, 0.75), 3), nsmall = 3),
      ", range ", format(round(min(cur_values), 3), nsmall = 3),
      " to ", format(round(max(cur_values), 3), nsmall = 3), "\n", sep = "")
}

for (cur_reason_column in c("not_measured_all", "not_measured_target")){

  cat("\nreasons ", cur_reason_column, " is set:\n", sep = "")

  reasons = df_result[get(cur_reason_column) != "", .N,
                      by = c(cur_reason_column)]

  if (nrow(reasons) == 0){
    cat("  none\n")
  }else{
    setorder(reasons, -N)
    for (cur_row in 1:nrow(reasons)){
      cat("  ", reasons$N[cur_row], ": ",
          reasons[[cur_reason_column]][cur_row], "\n", sep = "")
    }
  }
}

cat("\nimages left by a further floor on the pair, among those measured",
    " (nothing is filtered):\n", sep = "")
for (cur_floor in example_floors){
  cat("  n_pair >= ", cur_floor, ": ",
      sum(!is.na(df_result[[share_all_column]]) & df_result$n_pair >= cur_floor),
      " of ", sum(!is.na(df_result[[share_all_column]])),
      ",  n_pair_", target_ct_group, " >= ", cur_floor, ": ",
      sum(!is.na(df_result[[share_target_column]]) &
            df_result$n_pair_target >= cur_floor),
      " of ", sum(!is.na(df_result[[share_target_column]])), "\n", sep = "")
}

df_both = df_result[!is.na(get(share_all_column)) & !is.na(get(share_target_column))]

cat("\nboth measured in ", nrow(df_both), " images; Spearman ",
    format(round(cor(df_both[[share_all_column]], df_both[[share_target_column]],
                     method = "spearman"), 3), nsmall = 3),
    ", median absolute difference ",
    format(round(median(abs(df_both[[share_all_column]] -
                              df_both[[share_target_column]])), 3), nsmall = 3),
    "\n", sep = "")

cat("\nsaved ", output_csv, "\n", sep = "")
