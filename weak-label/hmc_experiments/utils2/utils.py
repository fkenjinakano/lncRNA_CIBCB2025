import pandas as pd
import numpy as np
import argparse
def load_datasets_inductive(path,
                  ): 
    dataset = pd.read_csv(path) ### datasets are preprocessed in a way that label columns are named like "label_"
    label_columns = np.array([True if "label" in l else False for l in dataset.columns ])
    x = dataset[dataset.columns[~label_columns]]
    y = dataset[dataset.columns[label_columns]]
    return x,y

def load_datasets_transductive(paths,
                  ): 
    
    dataset = pd.concat([pd.read_csv(p) for p in paths], axis=0) ### datasets are preprocessed in a way that label columns are named like "label_"
    label_columns = np.array([True if "label" in l else False for l in dataset.columns ])
    x = dataset[dataset.columns[~label_columns]]
    y = dataset[dataset.columns[label_columns]]
    return x,y

def filter_label_space(train_y,
                       test_y,
                       drop_labels_threshold = 30):
    if drop_labels_threshold > 0:
        train_y = train_y.loc[:, train_y.sum(axis=0) > drop_labels_threshold]
        test_y = test_y[train_y.columns]
    return train_y, test_y
def adjust_label_space_inductive(train_y,
                        test_y):
    reference_columns, candidate_columns = (train_y.columns, test_y.columns) if train_y.shape[1] > test_y.shape[1] else (test_y.columns, train_y.columns)
    labels_selected = [l for l in candidate_columns if l in reference_columns]    
    return train_y[labels_selected], test_y[labels_selected]
def adjust_label_space_transductive(train_y, 
                        test_y):
    return train_y.dropna(axis=1), test_y.dropna(axis=1)

       
def create_argparser_multiple_experiments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', nargs="+", type=int, choices = range(4))
    args = parser.parse_args()
    return args

# def create_argparser()
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-train",
#                      metavar="-tr", 
#                     help = "Path to the dataset (csv) used to train the model",
#                     required = True)
#     parser.add_argument("-valid",
#                     metavar="-va", 
#                     help = "Path to the dataset (csv) used to valid the model",
#                     required = True)

#     parser.add_argument("-test", 
#                     metavar = "-te", 
#                     help = "Path to the dataset (csv) used to test the model",
#                     required = True)
#     parser.add_argument("-impute",
#                             metavar = "--imp", 
#                             help = "Boolean flag for label imputation on the train set", 
#                             required = False,
#                             default = False)
#     parser.add_argument("-model",
#                             metavar = "--mo", 
#                             type = str, 
#                             help = "Model to be used", 
#                             choices = ["final_estimator", ## rf + et 
#                                        "lcforest",
#                                        "gcforest",
#                                        "cafe",
#                                        "cafe_os",
#                                        "cafe_slc",
#                                        "flaforest",
#                                        "cafe_fla"],
#                             required = True,
#                             )    
#     parser.add_argument("-mode",
#                             metavar = "--m", 
#                             type = str, 
#                             help = "Modus operandi: inductive or transductive", 
#                             choices = ["inductive", "transductive"],
#                             required = True,
#                             default = "inductive"
#                             )
#     parser.add_argument("-output",
#                             metavar = "--m", 
#                             type = str, 
#                             help = "Path of the output predictions file", 
#                             required = True,
#                             )    
#     args = parser.parse_args()
#     return args

def concatenate_datasets(*dataframes):
    concatenated_data = pd.concat(dataframes, axis=0).reset_index(drop=True)
    return concatenated_data