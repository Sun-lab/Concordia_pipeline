# a function providing a list of information for each dataset
# data_name


# group_scheme is LAST on purpose. Every caller of this function passes its
# arguments positionally, as data_features(data_name, model_run_id, data_dir),
# so a new parameter inserted before those two would silently re-bind them:
# model_run_id would arrive as group_scheme and data_dir as model_run_id.
# A new argument therefore goes on the end, or every call site changes with it.
data_features <- function(data_name, model_run_id=NULL, data_dir=NULL,
                          group_scheme="default"){
  
  # Lung cancer dataset
  if (data_name%in%c("cords_2024", "Cords_data_permutate", "Cords_data_permutate5pct")){
    
    cell_types = c('Bcell',
                   'Blood',
                   'CD4',
                   'CD4_Treg',
                   'CD8',
                   'Collagen_CAF',
                   'HEV',
                   'IDO_CAF',
                   'IDO_CD4',
                   'IDO_CD8',
                   'Lymphatic',
                   'Myeloid',
                   'Neutrophil',
                   'Other',
                   'PD1_CD4',
                   'PDPN_CAF',
                   'SMA_CAF',
                   'TCF1/7_CD4',
                   'TCF1/7_CD8',
                   'dCAF',
                   'hypoxic',
                   'hypoxic_CAF',
                   'hypoxic_tpCAF',
                   'iCAF',
                   'ki67_CD4',
                   'ki67_CD8',
                   'mCAF',
                   'normal',
                   'tpCAF',
                   'vCAF')
    
    stopifnot(group_scheme%in%c("default", "group8"))
    
    group_ct_mapping = list()
    
    if (group_scheme=="default"){
      
      group_ct_mapping[["immune"]] = c('Bcell',
                                       'CD4',
                                       'CD4_Treg',
                                       'CD8',
                                       'IDO_CD4',
                                       'IDO_CD8',
                                       'ki67_CD4',
                                       'ki67_CD8',
                                       'Myeloid',
                                       'Neutrophil',
                                       'PD1_CD4',
                                       'TCF1/7_CD4',
                                       'TCF1/7_CD8')
      
      group_ct_mapping[["tumor"]] = c('hypoxic', 
                                      'normal')
      
      group_ct_mapping[["Fibroblast"]] = c('Collagen_CAF',
                                           'dCAF',
                                           'hypoxic_CAF',
                                           'hypoxic_tpCAF',
                                           'iCAF',
                                           'IDO_CAF',
                                           'mCAF',
                                           'PDPN_CAF',
                                           'SMA_CAF',
                                           'tpCAF',
                                           'vCAF')
      
      group_ct_mapping[["vessel"]] = c('Blood', 
                                       'HEV', 
                                       'Lymphatic')
      
      group_ct_mapping[["Other"]] = c('Other')
      
    }else if(group_scheme=="group8"){
      
      group_ct_mapping[["immune"]] = c('Bcell',
                                       'CD4',
                                       'CD4_Treg',
                                       'CD8',
                                       'IDO_CD4',
                                       'IDO_CD8',
                                       'ki67_CD4',
                                       'ki67_CD8',
                                       'Myeloid',
                                       'Neutrophil',
                                       'PD1_CD4',
                                       'TCF1/7_CD4',
                                       'TCF1/7_CD8')
      
      group_ct_mapping[["tumor"]] = c('hypoxic', 
                                      'normal')
      
      group_ct_mapping[["caf1"]] = c('mCAF',
                                     'SMA_CAF')
      group_ct_mapping[["caf2"]] = c('Collagen_CAF',
                                     'hypoxic_CAF',
                                     'hypoxic_tpCAF',
                                     'PDPN_CAF')
      group_ct_mapping[["caf3"]] = c('iCAF',
                                     'IDO_CAF')
      group_ct_mapping[["caf4"]] = c('dCAF',
                                     'tpCAF',
                                     'vCAF')
      
      group_ct_mapping[["vessel"]] = c('Blood', 
                                       'HEV', 
                                       'Lymphatic')
      
      group_ct_mapping[["Other"]] = c('Other')
      
    }
      
    stopifnot(!is.null(data_dir))
    
    df_images = read.csv(file.path(data_dir, "region_list.csv"),
                         header=TRUE)
    
    raw_dir = file.path(data_dir, "raw_data")
    if (is.null(model_run_id)){
      result_subfolder = data_name
    }else{
      result_subfolder = model_run_id
    }
    
    n_cells_threshold = 30
    # the number of images threshold for deciding for each pair of clusters
    # whether to aggregate the distances across images
    # only aggregate if there are at least n_images_threshold images with 
    # at least n_cells_threshold in each of the two clusters
    n_images_threshold = 30
  }else if(data_name=="danenberg_2022"){
    
    cell_types = c('B cells',
                   'Basal',
                   'CD15^{+}',
                   'CD38^{+} lymphocytes',
                   'CD4^{+} T cells',
                   'CD4^{+} T cells & APCs',
                   'CD57^{+}',
                   'CD8^{+} T cells',
                   'CK^{+} CXCL12^{+}',
                   'CK^{lo}ER^{lo}',
                   'CK^{lo}ER^{med}',
                   'CK^{med}ER^{lo}',
                   'CK8-18^{+} ER^{hi}',
                   'CK8-18^{hi}CXCL12^{hi}',
                   'CK8-18^{hi}ER^{lo}',
                   'Endothelial',
                   'Ep CD57^{+}',
                   'Ep Ki67^{+}',
                   'ER^{hi}CXCL12^{+}',
                   'Fibroblasts',
                   'Fibroblasts FSP1^{+}',
                   'Granulocytes',
                   'HER2^{+}',
                   'Ki67^{+}',
                   'Macrophages',
                   'Macrophages & granulocytes',
                   'MHC I & II^{hi}',
                   'MHC I^{hi}CD57^{+}',
                   'MHC^{hi}CD15^{+}',
                   'Myofibroblasts',
                   'Myofibroblasts PDPN^{+}',
                   'T_{Reg} & T_{Ex}')    
    
    stopifnot(group_scheme%in%c("default"))
    
    group_ct_mapping = list()
    
    if (group_scheme=="default"){
      
      group_ct_mapping[["immune"]] = c('T_{Reg} & T_{Ex}',
                                       'CD4^{+} T cells & APCs',
                                       'CD4^{+} T cells',
                                       'CD8^{+} T cells',
                                       'B cells',
                                       'CD38^{+} lymphocytes',
                                       'Granulocytes',
                                       'Macrophages',
                                       'Macrophages & granulocytes',
                                       'Ki67^{+}',
                                       'CD57^{+}')
      
      group_ct_mapping[["tumor"]] = c('MHC I & II^{hi}',
                                      'MHC I^{hi}CD57^{+}',
                                      'MHC^{hi}CD15^{+}',
                                      'HER2^{+}',
                                      'CK8-18^{+} ER^{hi}',
                                      'CK^{lo}ER^{med}',
                                      'Basal',
                                      'ER^{hi}CXCL12^{+}',
                                      'CK8-18^{hi}CXCL12^{hi}',
                                      'CD15^{+}',
                                      'Ep CD57^{+}',
                                      'CK^{+} CXCL12^{+}',
                                      'Ep Ki67^{+}',
                                      'CK^{med}ER^{lo}',
                                      'CK^{lo}ER^{lo}',
                                      'CK8-18^{hi}ER^{lo}')
      
      group_ct_mapping[["stroma"]] = c('Fibroblasts',
                                       'Fibroblasts FSP1^{+}',
                                       'Myofibroblasts',
                                       'Myofibroblasts PDPN^{+}')
      
      group_ct_mapping[["endothelial"]] = c('Endothelial')
      
    }
    
    stopifnot(!is.null(data_dir))
    
    df_images_info = read.csv(file.path(data_dir, "Metabric_images_geq_1000_cells.csv"),
                              header=TRUE)
    
    full_subtype = c("Positive", "Negative")
    df_images_subtype = df_images_info[which(df_images_info$er_sd%in%full_subtype),]
    df_images = data.frame(region_ID=df_images_subtype$image_fullname)
    
    raw_dir = file.path(data_dir, "raw_data")
    if (is.null(model_run_id)){
      result_subfolder = data_name
    }else{
      result_subfolder = model_run_id
    }
    
    n_cells_threshold = 20
    # the number of images threshold for deciding for each pair of clusters
    # whether to aggregate the distances across images
    # only aggregate if there are at least n_images_threshold images with 
    # at least n_cells_threshold in each of the two clusters
    n_images_threshold = 10
    
  }else if(data_name%in%c("mpfc_2022")){
    
    cell_types = c("Smc", "Endo", "Astro", "eL6-2", "Oligo",
                   "eL5-3", "Reln", "VIP", "eL2/3", "NPY", 
                   "eL5-2", "Lhx6", "L5-1", "SST", "eL6-1")
    
    # the same check the cords branch makes, and for the same reason: without it
    # an unrecognised scheme leaves group_ct_mapping the empty list it was
    # initialised to and the function returns successfully with no mapping at all.
    # Only "default" exists here, one group per cell type; "group8" is a grouping
    # of the Cords cell types and has no meaning for this data
    stopifnot(group_scheme %in% c("default"))
    
    group_ct_mapping = list()
    
    if (group_scheme=="default"){
      for (ct in cell_types){
        group_ct_mapping[[ct]] = c(ct)
      }
    }
    
    if (is.null(data_dir)){
      data_dir = "../../data/mpfc_data"
    }
    
    df_images_info_full = read.csv(file.path(data_dir, "image_info.csv"),
                                   header=TRUE)
    df_images_info = df_images_info_full[df_images_info_full$subtype=="all",]
    df_images = data.frame(region_ID=df_images_info$image_fullname)
    
    raw_dir = file.path(data_dir, "raw_data")
    if (is.null(model_run_id)){
      result_subfolder = data_name
    }else{
      result_subfolder = model_run_id
    }
    
    n_cells_threshold = 10
    # the number of images threshold for deciding for each pair of clusters
    # whether to aggregate the distances across images
    # only aggregate if there are at least n_images_threshold images with 
    # at least n_cells_threshold in each of the two clusters
    n_images_threshold = 1
  
  }
  
  return_list = list("cell_types" = cell_types, 
                     "group_ct_mapping" = group_ct_mapping, 
                     "df" = df_images, 
                     "raw_dir"=raw_dir, 
                     "result_subfolder"=result_subfolder, 
                     "n_cells_threshold"=n_cells_threshold, 
                     "n_images_threshold"=n_images_threshold)
  
  return(return_list)
  
}
