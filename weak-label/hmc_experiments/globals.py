PATH_2007 = "datasets/2007" 
PATH_2018 = "datasets/2018"

DATASETS = [
            "cellcycle_FUN",
            "derisi_FUN",
            "eisen_FUN",
#            "expr_FUN",
            "gasch1_FUN",
            "gasch2_FUN",
            "seq_FUN",
            "spo_FUN"
            ]
# MODELS = [
#     "lcforest"
#     "gcforest"
#     "cafe"
#     "cafe_os"
#     "cafe_slc"
#     "flaforest"
#     "cafe_fla"
#     "final_estimator"
#     ]

MODELS = [
#    "cale",
#    "cale_os",
#    "lcforest",
    "gcforest",
#    "cafe", ## tree-embeddings, no label imputation
#    "cafe_os", ## output features + tree_embeddings, no label imputation
#    "cafe_slc", ## tree-embeddings, output features, with SLC label imputation
#    "flaforest",
#    "cafe_fla", ## tree-embeddings, output features, with FLA label imputation
#    "final_estimator", ## rf + et
    ]
