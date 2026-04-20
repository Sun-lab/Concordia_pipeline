import pandas as pd
from collections import defaultdict

class data_features(object):
    def __init__(self, data_name, subtype, graph_type):

        if graph_type=="extended":
            self.processed_folder_name = "tg_graph"
        elif graph_type=="basic":
            self.processed_folder_name = "tg_graph_basic"
        elif graph_type=="1st":
            self.processed_folder_name = "tg_graph_1st"
        elif graph_type=="2nd":
            self.processed_folder_name = "tg_graph_2nd"

        if "cords" in data_name:

            self.raw_dir = "../data/Cords_data/raw_data"

            self.cell_type_mapping = {'Bcell': 0,
                                        'Blood': 1,
                                        'CD4': 2,
                                        'CD4_Treg': 3,
                                        'CD8': 4,
                                        'Collagen_CAF': 5,
                                        'HEV': 6,
                                        'IDO_CAF': 7,
                                        'IDO_CD4': 8,
                                        'IDO_CD8': 9,
                                        'Lymphatic': 10,
                                        'Myeloid': 11,
                                        'Neutrophil': 12,
                                        'Other': 13,
                                        'PD1_CD4': 14,
                                        'PDPN_CAF': 15,
                                        'SMA_CAF': 16,
                                        'TCF1/7_CD4': 17,
                                        'TCF1/7_CD8': 18,
                                        'dCAF': 19,
                                        'hypoxic': 20,
                                        'hypoxic_CAF': 21,
                                        'hypoxic_tpCAF': 22,
                                        'iCAF': 23,
                                        'ki67_CD4': 24,
                                        'ki67_CD8': 25,
                                        'mCAF': 26,
                                        'normal': 27,
                                        'tpCAF': 28,
                                        'vCAF': 29}


            self.group_ct_mapping = defaultdict(set)

            if graph_type in ["extended", "basic", "1st", "2nd"]:

                self.dataset_root = "../data/Cords_data/graph_objects_degree_20"

                self.group_ct_mapping["immune"] = set(['Bcell',
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
                                                    'TCF1/7_CD8'])

                self.group_ct_mapping["tumor"] = set(['hypoxic',
                                                    'normal'])

                self.group_ct_mapping["Fibroblast"] = set(['Collagen_CAF',
                                                            'dCAF',
                                                            'hypoxic_CAF',
                                                            'hypoxic_tpCAF',
                                                            'iCAF',
                                                            'IDO_CAF',
                                                            'mCAF',
                                                            'PDPN_CAF',
                                                            'SMA_CAF',
                                                            'tpCAF',
                                                            'vCAF'])

                self.group_ct_mapping["vessel"] = set(['Blood',
                                                        'HEV',
                                                        'Lymphatic'])

                self.group_ct_mapping["Other"] = set(['Other'])

            df_images = pd.read_csv("../"+ \
                                    "data/Cords_data/patient_image_greq_1000_cells.csv",
                                    header=0)

            if subtype=="LUAD":
                full_subtype = ["Adenocarcinoma"]
            elif subtype=="LUSC":
                full_subtype = ["Squamous cell carcinoma"]
            elif subtype=="both":
                full_subtype = ["Adenocarcinoma", "Squamous cell carcinoma"]


            df_images_subtype = df_images[df_images['DX.name'].isin(full_subtype)]
            all_patients = list(set(df_images_subtype['Patient_ID']))
            all_patients.sort()
            self.patients = all_patients

            train_images_raw = df_images_subtype["RoiID"].tolist()
            self.train_images = ["_".join(x.split(",")) for x in train_images_raw]

            self.dist_cutoff = 16
            self.path_purity_cutoff = 0.90
            self.path_len_cutoff = 30000

            self.data_subfolder = "cords_2024"
            self.n_cells_threshold = 30

        elif "danenberg" in data_name:

            self.raw_dir = "../data/Danenberg_data/raw_data"
            self.dataset_root = "../data/Danenberg_data/graph_objects_degree_20"

            self.cell_type_mapping = {'B cells': 0,
                                        'Basal': 1,
                                        'CD15^{+}': 2,
                                        'CD38^{+} lymphocytes': 3,
                                        'CD4^{+} T cells': 4,
                                        'CD4^{+} T cells & APCs': 5,
                                        'CD57^{+}': 6,
                                        'CD8^{+} T cells': 7,
                                        'CK^{+} CXCL12^{+}': 8,
                                        'CK^{lo}ER^{lo}': 9,
                                        'CK^{lo}ER^{med}': 10,
                                        'CK^{med}ER^{lo}': 11,
                                        'CK8-18^{+} ER^{hi}': 12,
                                        'CK8-18^{hi}CXCL12^{hi}': 13,
                                        'CK8-18^{hi}ER^{lo}': 14,
                                        'Endothelial': 15,
                                        'Ep CD57^{+}': 16,
                                        'Ep Ki67^{+}': 17,
                                        'ER^{hi}CXCL12^{+}': 18,
                                        'Fibroblasts': 19,
                                        'Fibroblasts FSP1^{+}': 20,
                                        'Granulocytes': 21,
                                        'HER2^{+}': 22,
                                        'Ki67^{+}': 23,
                                        'Macrophages': 24,
                                        'Macrophages & granulocytes': 25,
                                        'MHC I & II^{hi}': 26,
                                        'MHC I^{hi}CD57^{+}': 27,
                                        'MHC^{hi}CD15^{+}': 28,
                                        'Myofibroblasts': 29,
                                        'Myofibroblasts PDPN^{+}': 30,
                                        'T_{Reg} & T_{Ex}': 31
                                    }

            assert graph_type in ["extended", "basic", "1st", "2nd"]

            if graph_type in ["extended", "basic", "1st", "2nd"]:

                self.group_ct_mapping = defaultdict(set)

                self.group_ct_mapping["immune"] = set(['T_{Reg} & T_{Ex}',
                                                    'CD4^{+} T cells & APCs',
                                                    'CD4^{+} T cells',
                                                    'CD8^{+} T cells',
                                                    'B cells',
                                                    'CD38^{+} lymphocytes',
                                                    'Granulocytes',
                                                    'Macrophages',
                                                    'Macrophages & granulocytes',
                                                    'Ki67^{+}',
                                                    'CD57^{+}'])

                self.group_ct_mapping["tumor"] = set(['MHC I & II^{hi}',
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
                                                    'CK8-18^{hi}ER^{lo}'])

                self.group_ct_mapping["stroma"] = set(['Fibroblasts',
                                                    'Fibroblasts FSP1^{+}',
                                                    'Myofibroblasts',
                                                    'Myofibroblasts PDPN^{+}'])

                self.group_ct_mapping["endothelial"] = set(['Endothelial'])


            df_images = pd.read_csv("../"+ \
                                    "data/Danenberg_data/Metabric_images_geq_1000_cells.csv",
                                    header=0)

            if subtype=="ERpos":
                full_subtype = ["Positive"]
            elif subtype=="ERneg":
                full_subtype = ["Negative"]
            elif subtype=="both":
                full_subtype = ["Positive", "Negative"]

            df_images_subtype = df_images[df_images['er_sd'].isin(full_subtype)]
            all_patients = list(set(df_images_subtype['metabric_id']))
            all_patients.sort()
            self.patients = all_patients

            self.train_images = df_images_subtype["image_fullname"].tolist()

            self.dist_cutoff = 20
            self.path_purity_cutoff = 0.90
            self.path_len_cutoff = 100000

            self.data_subfolder = "danenberg_2022"
            self.n_cells_threshold = 20

        elif "mpfc" in data_name:

            self.raw_dir = "../data/mpfc_data/raw_data"
            self.dataset_root = "../data/mpfc_data/graph_objects_degree_20"

            all_celltypes = ["Smc", "Endo", "Astro", "eL6-2", "Oligo",
                             "eL5-3", "Reln", "VIP", "eL2/3", "NPY", 
                             "eL5-2", "Lhx6", "L5-1", "SST", "eL6-1"]
            all_celltypes.sort()

            self.cell_type_mapping = {}

            for i, x in enumerate(all_celltypes):
                self.cell_type_mapping[x] = i

            assert graph_type in ["extended", "basic", "1st", "2nd"]

            if graph_type in ["extended", "basic", "1st", "2nd"]:

                self.group_ct_mapping = defaultdict(set)

                for ct in all_celltypes:
                    self.group_ct_mapping[ct] = set([ct])


            df_images = pd.read_csv("../"+ \
                                    "data/mpfc_data/image_info.csv",
                                    header=0)

            if subtype=="all":
                full_subtype = ["all"]

            df_images_subtype = df_images[df_images['subtype'].isin(full_subtype)]
            all_patients = list(set(df_images_subtype['patient_id']))
            all_patients.sort()
            self.patients = all_patients

            self.train_images = df_images_subtype["image_fullname"].tolist()

            self.dist_cutoff = 380
            self.path_purity_cutoff = 0.90
            self.path_len_cutoff = 100000

            self.data_subfolder = "mpfc_2022"
            self.n_cells_threshold = 10