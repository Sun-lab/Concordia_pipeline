# after filtering out ROI with low cell counts (<1000), for ROI with kept patients, 
# output two types of files:
# cell information file per ROI
# patient information file


library(SingleCellExperiment)
library(dplyr)
library(ggplot2)
library(ggpubr)
library(viridis)
library(ggpointdensity)

# ---------------------------------------------
# load data
# ---------------------------------------------

ori_path = "../downloaded_data/Cords_2024"
sc_object <- readRDS(file.path(ori_path, "SingleCellExperiment_Objects/sce_all_annotated.rds"))
sc_object

table(colData(sc_object)$cell_type)
length(unique(colData(sc_object)$cell_type))

colnames(colData(sc_object))
table(colData(sc_object)$cell_category)

# there are two items possibly for cells with no clearly identified cell types
# "other" and "Other", may combine them into one in later steps
table(colData(sc_object)$cell_subtype)
length(unique(colData(sc_object)$cell_subtype))

table(colData(sc_object)$cell_type, 
      colData(sc_object)$cell_subtype)

table(colData(sc_object)$cell_subtype, 
      colData(sc_object)$cell_category)

length(unique(colData(sc_object)$Patient_ID))
unique(colData(sc_object)$Patient_ID)[1:10]
unique(colData(sc_object)$Patient_Nr)[1:10]

colData(sc_object)$ROI_xy[1:10]
colData(sc_object)$Center_X[1:10]
colData(sc_object)$Center_Y[1:10]
summary(colData(sc_object)$Center_X)
summary(colData(sc_object)$Center_Y)
# these summaries on area in mm and area in px of the cores have duplicates by the number of cells
summary(colData(sc_object)$Area_mm_Core) # if doing scatterplot, should do it on image level
summary(colData(sc_object)$Area_px_Core)


# read the data frame of patients under consideration
df_patient_w_response_DX_kept_no_NeoAdj = read.csv("../data/Cords_data/patient_roi_w_response_DX_kept_no_NeoAdj.csv", 
                                                   header = TRUE)
dim(df_patient_w_response_DX_kept_no_NeoAdj)
head(df_patient_w_response_DX_kept_no_NeoAdj)


# ---------------------------------------------
# filter out problematic ROI IDs
# ---------------------------------------------

# consistency among the area in mm, area in px, and number of cells

df_roi = unique(colData(sc_object)[, c("RoiID", "Area_mm_Core","Area_px_Core")])
dim(df_roi)
length(unique(df_roi$RoiID))

# see which ROI has nonunique information

roi_duplicate = data.frame(table(df_roi$RoiID))
troublesome_rois = roi_duplicate[which(roi_duplicate$Freq>1), ]$Var1

# there are ROI IDs with nonunique values for Area_mm_Core and Area_px_Core
# see what kind of patients they come from

df_cell_for_troublesome_rois = colData(sc_object)[which(colData(sc_object)$RoiID==troublesome_rois[1]), ]
dim(df_cell_for_troublesome_rois)
table(df_cell_for_troublesome_rois$Patient_ID, useNA="ifany")
table(df_cell_for_troublesome_rois$DX.name, useNA="ifany")
table(df_cell_for_troublesome_rois$NeoAdj, useNA="ifany")
table(df_cell_for_troublesome_rois$Area_mm_Core)
table(df_cell_for_troublesome_rois$Area_px_Core)
length(unique(df_cell_for_troublesome_rois$CellNumber))
unique(colData(sc_object)[which(colData(sc_object)$Patient_ID==df_cell_for_troublesome_rois$Patient_ID[1]), ]$RoiID)

df_cell_for_troublesome_rois = colData(sc_object)[which(colData(sc_object)$RoiID==troublesome_rois[2]), ]
dim(df_cell_for_troublesome_rois)
table(df_cell_for_troublesome_rois$Patient_ID, useNA="ifany")
table(df_cell_for_troublesome_rois$DX.name, useNA="ifany")
table(df_cell_for_troublesome_rois$NeoAdj, useNA="ifany")
table(df_cell_for_troublesome_rois$Area_mm_Core)
table(df_cell_for_troublesome_rois$Area_px_Core)
length(unique(df_cell_for_troublesome_rois$CellNumber))
unique(colData(sc_object)[which(colData(sc_object)$Patient_ID==df_cell_for_troublesome_rois$Patient_ID[2]), ]$RoiID)


# exclude these two images from the ROI to use

df_roi_to_plot = df_roi[which(!df_roi$RoiID%in%troublesome_rois), ]
dim(df_roi)
dim(df_roi_to_plot)


# ------------------------------------------
# explore ROI properties
# ------------------------------------------

summary(df_roi_to_plot$Area_px_Core/df_roi_to_plot$Area_mm_Core)

df_cell_info = as.data.frame(colData(sc_object))

df_unique_roi = df_cell_info %>%
  group_by(RoiID) %>%
  summarise(n_cell = n(), 
            n_unique_cell = n_distinct(CellNumber))

dim(df_unique_roi)
head(df_unique_roi)

table(df_unique_roi$n_cell==df_unique_roi$n_unique_cell, useNA="ifany")
df_unique_roi[which(df_unique_roi$n_cell!=df_unique_roi$n_unique_cell),]

df_unique_ori_matched = df_unique_roi[match(df_roi_to_plot$RoiID, df_unique_roi$RoiID),]
dim(df_unique_ori_matched)

table(df_roi_to_plot$RoiID==df_unique_ori_matched$RoiID, useNA="ifany")

df_roi_to_plot$n_cells = df_unique_ori_matched$n_cell

table(df_roi_to_plot$n_cells < 200)
table(df_roi_to_plot$n_cells < 400)
table(df_roi_to_plot$n_cells < 600)
table(df_roi_to_plot$n_cells < 800)
table(df_roi_to_plot$n_cells < 1000)

p_list = list()

p_list[[1]] = ggplot(df_roi_to_plot, aes(x=Area_mm_Core))+
              geom_histogram(color="darkblue", fill="lightblue") + 
              theme_classic()
p_list[[2]] = ggplot(df_roi_to_plot, aes(x=n_cells))+
              geom_histogram(color="darkblue", fill="lightblue") + 
              theme_classic() + geom_vline(xintercept=200, col = "grey") + 
              geom_vline(xintercept=400, col = "grey") + 
              geom_vline(xintercept=600, col = "grey") + 
              geom_vline(xintercept=800, col = "grey") + 
              geom_vline(xintercept=1000, col = "grey")
p_list[[3]] = ggplot(df_roi_to_plot, aes(x=Area_mm_Core, y=n_cells)) +
              geom_point() + theme_classic() + 
              geom_pointdensity() +
              scale_color_viridis() + theme_classic() + 
              geom_hline(yintercept=200, col = "grey") + 
              geom_hline(yintercept=400, col = "grey") + 
              geom_hline(yintercept=600, col = "grey") + 
              geom_hline(yintercept=800, col = "grey") + 
              geom_hline(yintercept=1000, col = "grey")



# -----------------------------------------------
# filter out any ROI with fewer than 1000 cells
# -----------------------------------------------


dim(df_patient_w_response_DX_kept_no_NeoAdj)
length(unique(df_patient_w_response_DX_kept_no_NeoAdj$Patient_ID))
length(unique(df_patient_w_response_DX_kept_no_NeoAdj$RoiID))

# save a file with patient ROI and label information
head(df_patient_w_response_DX_kept_no_NeoAdj)

df_patient_w_response_DX_kept_no_NeoAdj_sorted <- 
  df_patient_w_response_DX_kept_no_NeoAdj[order(df_patient_w_response_DX_kept_no_NeoAdj$Patient_ID),]
head(df_patient_w_response_DX_kept_no_NeoAdj_sorted)

table(df_patient_w_response_DX_kept_no_NeoAdj_sorted$RoiID%in%df_roi_to_plot$RoiID, useNA="ifany")

dim(df_patient_w_response_DX_kept_no_NeoAdj_sorted)

df_patient_w_response_DX_kept_no_NeoAdj_sorted_rm_two = 
  df_patient_w_response_DX_kept_no_NeoAdj_sorted[which(!df_patient_w_response_DX_kept_no_NeoAdj_sorted$RoiID%in%troublesome_rois),]

dim(df_patient_w_response_DX_kept_no_NeoAdj_sorted_rm_two)

df_roi_to_plot_matched = df_roi_to_plot[match(df_patient_w_response_DX_kept_no_NeoAdj_sorted_rm_two$RoiID, 
                                              df_roi_to_plot$RoiID),]

table(df_patient_w_response_DX_kept_no_NeoAdj_sorted_rm_two$RoiID == df_roi_to_plot_matched$RoiID, 
      useNA="ifany")


df_patient_w_response_DX_kept_no_NeoAdj_sorted_rm_two$n_cells = df_roi_to_plot_matched$n_cells
head(df_patient_w_response_DX_kept_no_NeoAdj_sorted_rm_two)

df_patient_w_response_DX_kept_no_NeoAdj_sorted_rm_two$response = 
  factor(df_patient_w_response_DX_kept_no_NeoAdj_sorted_rm_two$response)

wilcox_pvalue = round(wilcox.test(n_cells ~ response, data = df_patient_w_response_DX_kept_no_NeoAdj_sorted_rm_two,
                                 exact = FALSE)$p.value, digits =5)
wilcox_pvalue

p_list[[4]] = ggplot() + theme_void()
p_list[[5]] = ggplot(df_patient_w_response_DX_kept_no_NeoAdj_sorted_rm_two, 
                     aes(x=response, y=n_cells, color=response)) +
              geom_boxplot() + theme_classic() + 
              ggtitle(paste0("Before flitering out ROIs with n_cells < 1000", 
                              "\nno separation between LUADs and LUSCs\nWilcox pvalue: ", as.character(wilcox_pvalue)))
  


# filter out ROI with fewer than 1000 cells
table(df_patient_w_response_DX_kept_no_NeoAdj_sorted_rm_two$n_cells<1000)
df_output = df_patient_w_response_DX_kept_no_NeoAdj_sorted_rm_two[which(df_patient_w_response_DX_kept_no_NeoAdj_sorted_rm_two$n_cells>=1000),]
dim(df_output)

wilcox_pvalue_filtered = round(wilcox.test(n_cells ~ response, data = df_output,
                               exact = FALSE)$p.value, digits =5)
wilcox_pvalue_filtered

# generate exploration figures

p_list[[6]] = ggplot(df_output, 
                     aes(x=response, y=n_cells, color=response)) +
              geom_boxplot() + theme_classic() + 
              ggtitle(paste0("After flitering out ROIs with n_cells < 1000", 
                             "\nno separation between LUADs and LUSCs\nWilcox pvalue: ", 
                             as.character(wilcox_pvalue_filtered)))

pdf(file = paste0("../figures/cords_2024/explore_roi_info.pdf"), 
    width = 8.6, height =9)
ggarrange(plotlist = p_list, ncol = 2, nrow = 3)
dev.off()


write.csv(df_output, 
          "../data/Cords_data/patient_image_greq_1000_cells.csv", 
          row.names=FALSE)

table(df_output$response, useNA="ifany")

df_output_patients = unique(df_output[, c("Patient_ID", "DX.name", "NeoAdj", "response")])
dim(df_output_patients)

table(df_output_patients$response, useNA="ifany")

table(df_output_patients$DX.name, 
      df_output_patients$response)

write.csv(df_output_patients, 
          "../data/Cords_data/Cords_patients_reponse.csv", 
          row.names=FALSE)
# -----------------------------------------------
# write out chosen ROIs
# -----------------------------------------------


dim(df_cell_info)
head(df_cell_info)

length(unique(df_cell_info$CellNumber))

df_cell_info$CELL_ID = 1:nrow(df_cell_info)
df_cell_key_info = df_cell_info[, c("CELL_ID", "Center_X", "Center_Y", 
                                    "cell_subtype", "Patient_ID", "RoiID")]
head(df_cell_key_info)
table(df_cell_key_info$cell_subtype, useNA="ifany")
colnames(df_cell_key_info)[2] = "X"
colnames(df_cell_key_info)[3] = "Y"
colnames(df_cell_key_info)[4] = "CELL_TYPE"

cell_key_info_ct_column = df_cell_key_info$CELL_TYPE
cell_key_info_ct_column[which(cell_key_info_ct_column=="other")] = "Other"
df_cell_key_info$CELL_TYPE = cell_key_info_ct_column

for (i in 1:nrow(df_output)){
    
  cur_roi = df_output$RoiID[i]
  format_cur_roi = paste(unlist(strsplit(cur_roi, ",")), collapse = "_")
  
  df_cell_cur = df_cell_key_info[which(df_cell_key_info$RoiID==cur_roi),]
  stopifnot(nrow(df_cell_cur)==length(unique(df_cell_cur$CELL_ID)))
  stopifnot(length(unique(df_cell_cur$Patient_ID))==1)  
  
  write.csv(df_cell_cur, 
            file = paste0("../data/Cords_data/raw_data/", format_cur_roi, ".csv"), 
            row.names=FALSE)
  
  if (i%%10==0){
    print(paste0("done with image with index: ", i))
    print(Sys.time())
  }
}

dim(df_output)


sessionInfo()
q(save="no")



