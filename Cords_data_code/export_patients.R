# export raw patient and roi table
# to help with deciding rois to use and patient feature table in later steps

library(SingleCellExperiment)
library(dplyr)

ori_path = "../downloaded_data/Cords_2024"
sc_object <- readRDS(file.path(ori_path, "SingleCellExperiment_Objects/sce_all_annotated.rds"))
sc_object

dim(colData(sc_object))
colnames(colData(sc_object))

table(colData(sc_object)$cell_type)

length(unique(colData(sc_object)$cell_type))

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


unique(colData(sc_object)$Relapse)
table(colData(sc_object)$Relapse, useNA="ifany")

unique(colData(sc_object)$Typ)

unique(colData(sc_object)$BatchID)

head(colData(sc_object))

unique(colData(sc_object)$Panel)
unique(colData(sc_object)$DX.name)
unique(colData(sc_object)$NeoAdj)

# column RoiID seems to be the ID for region of interest
length(unique(colData(sc_object)$ImageNumber))
length(unique(colData(sc_object)$TmaID))
length(unique(colData(sc_object)$RoiID))
length(unique(colData(sc_object)$ROI))

df_patient_response = unique(colData(sc_object)[, c("Patient_ID", "Relapse", "DX.name", "NeoAdj", "RoiID")])
dim(df_patient_response)

length(unique(df_patient_response$Patient_ID))
length(unique(df_patient_response$RoiID))

table(df_patient_response$DX.name)
length(unique(df_patient_response[which(df_patient_response$DX.name=="Control"),]$Patient_ID))
unique(df_patient_response[which(df_patient_response$DX.name=="Control"),]$Patient_ID)
length(unique(df_patient_response[which(df_patient_response$DX.name=="Control"),]$RoiID))

# align with the claim in the paper that
# The samples included 618 pathologist-classified LUADs and 401 LUSCs
length(unique(df_patient_response[which(df_patient_response$DX.name=="Adenocarcinoma"), ]$Patient_ID))
length(unique(df_patient_response[which(df_patient_response$DX.name=="Squamous cell carcinoma"), ]$Patient_ID))

# these counts are not on unique patient level, 
# since there are duplicates and one patient can have multiple rows
# look into whether the duplicates come from
table(df_patient_response$Relapse, useNA="ifany")
table(df_patient_response$DX.name, useNA="ifany")
table(df_patient_response$NeoAdj, useNA="ifany")

df_patient = as.data.frame(df_patient_response)

df_patient_n_unique = df_patient %>%
  group_by(Patient_ID) %>%
  summarise(n_DX.name = n_distinct(DX.name), 
            n_NeoAdj = n_distinct(NeoAdj), 
            n_Relapse = n_distinct(Relapse), 
            n_images = n_distinct(RoiID))

dim(df_patient_n_unique)

df_patient_n_unique[which(df_patient_n_unique$n_DX.name>1),]
df_patient_n_unique[which(df_patient_n_unique$n_NeoAdj>1),]
df_patient_n_unique[which(df_patient_n_unique$n_Relapse>1),]

# look into the 5 patients with more than one unique values in DX.name
# it turns out to be due to different ways of coding NA
double_DX_patients = df_patient_n_unique[which(df_patient_n_unique$n_DX.name>1),]$Patient_ID
df_patient_double_DX = df_patient[which(df_patient$Patient_ID%in%double_DX_patients),]
df_patient_double_DX[order(df_patient_double_DX$Patient_ID),]

table(df_patient$DX.name, useNA = "ifany")

# replace the original 'NA' string with real NA in R
df_patient$DX.name[which(df_patient$DX.name=="NA")] = NA

table(df_patient$Relapse, useNA="ifany")

df_unique_patients = df_patient %>%
                    group_by(Patient_ID) %>%
                    summarise(n_DX.name = n_distinct(DX.name), 
                              n_NeoAdj = n_distinct(NeoAdj), 
                              n_Relapse = n_distinct(Relapse), 
                              n_images = n_distinct(RoiID))

# the number of patients with 1 or 2 images align with what is mentioned in the paper
# as on page 3 of the paper, 
# "We analyzed two cores for 85% (n = 909) and one core for 15% (n = 161) of patients"
table(df_unique_patients$n_images, useNA="ifany")

# the patient corresponding to 91 images is the control patient
df_unique_patients[which(df_unique_patients$n_images==91),]

df_patient_w_response = df_patient[which(!is.na(df_patient$Relapse)), ]
table(df_patient_w_response$Relapse, useNA="ifany")

# consistent with the statement in the paper that 480 patients had relapses within the 15 years of follow-up
df_pure_patient_response = unique(df_patient_w_response[, c("Patient_ID","Relapse", "DX.name")])
dim(df_pure_patient_response)
length(unique(df_pure_patient_response$Patient_ID))
table(df_pure_patient_response$Relapse)

patients_w_response = df_pure_patient_response$Patient_ID
length(patients_w_response)

# there are only 80 unique patient IDs with NeoAdj=="NeoAdjuvantTherapy"
# this is not consistent with the number 82 mentioned in the paper
table(colData(sc_object)$NeoAdj, useNA="ifany")
length(unique(colData(sc_object)[which(colData(sc_object)$NeoAdj=="NeoAdjuvantTherapy"), ]$Patient_ID))

table(df_unique_patients$n_NeoAdj)
table(df_patient$NeoAdj)
length(unique(df_patient[which(df_patient$NeoAdj=="NeoAdjuvantTherapy"),]$Patient_ID))


# select the patients to use
# be either among the 618 pathologist-classified LUADs or 401 LUSCs
# exclude the 80 (inconsistent with the 82 stated in the paper) patients who received neoadjuvant therapy
# have relapse or not status
# make sure to exclude the one for normal samples

# the patients without relapse or not response are already filtered out in the processing of getting to df_patient_w_response 
# exclude those with DX.name not in c("Adenocarcinoma", "Squamous cell carcinoma")
df_patient_w_response_DX_kept = df_patient_w_response[which(df_patient_w_response$DX.name%in%c("Adenocarcinoma", "Squamous cell carcinoma")), ]
dim(df_patient_w_response_DX_kept)
# exclude the patients who received neoadjuvant therapy
df_patient_w_response_DX_kept_no_NeoAdj = df_patient_w_response_DX_kept[which(df_patient_w_response_DX_kept$NeoAdj=="NoNeoAdjuvantTherapy"), ]
dim(df_patient_w_response_DX_kept_no_NeoAdj)

head(df_patient_w_response_DX_kept_no_NeoAdj)

length(unique(df_patient_w_response_DX_kept_no_NeoAdj$Patient_ID))

# create a new column called response, with value 1 for no relapse within 15 years
summary(df_patient_w_response_DX_kept_no_NeoAdj$Relapse)
df_patient_w_response_DX_kept_no_NeoAdj$response = 1 - df_patient_w_response_DX_kept_no_NeoAdj$Relapse
summary(df_patient_w_response_DX_kept_no_NeoAdj$response)

df_patients_to_use = unique(df_patient_w_response_DX_kept_no_NeoAdj[, c("Patient_ID", "Relapse", "response", "DX.name", "NeoAdj")])
dim(df_patients_to_use)
head(df_patients_to_use)

table(df_patients_to_use$Relapse, useNA="ifany")
table(df_patients_to_use$response, useNA="ifany")
table(df_patients_to_use$DX.name, useNA="ifany")
table(df_patients_to_use$NeoAdj, useNA="ifany")

# save the data frame with RoiID information out
write.csv(df_patient_w_response_DX_kept_no_NeoAdj, 
          file = "../data/Cords_data/patient_roi_w_response_DX_kept_no_NeoAdj.csv", 
          row.names = FALSE)

# the step of further filtering images by cell number is left a new step


q(save="no")

sessionInfo()



