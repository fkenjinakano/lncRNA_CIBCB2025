from models.estimators import estimators_dict
import numpy as np
import sys
import pandas as pd
import copy
sys.path.append("../../")
from utils.utils import iterator_cross_validation
from deepforest import DeepForest

if __name__ == "__main__":
    '''estimators_dict = {
        "lcforest": lcforest,
        "gcforest": gcforest,
        "cafe": cafe,
        "cafe_os": cafe_os,
        "cafe_slc": cafe_slc,
        "flaforest": flaforest,
        "cafe_fla": cafe_fla,
        "cale": cale,
        "cale_os": cale_os,
        "final_estimator": final_estimator,
    }'''
    path_features = sys.argv[1]
    path_labels = sys.argv[2]
    path_indexes = sys.argv[3]
    model_index = int(sys.argv[4])
    binary_mode = sys.argv[5] == "true"
    if binary_mode:
        disease_similarities = pd.read_csv(sys.argv[6])
    else:
        disease_similarities = None
    transductive_mode = False
    indexes_to_mask = None
    concatenate_kmers_with_svd = False
    ### RUNNING MODELS 1, 2, 3,  4, 6, 9
    models = ['lcforest', #0
            'gcforest', #1
            'cafe', #2
            'cafe_os', #3 
            'cafe_slc', #4
            'flaforest', #5
            'cafe_fla', #6
            'cale', #7
            'cale_os', #8
            'final_estimator' #9
             ]
    model_name = models[model_index]

    x = pd.read_csv(path_features, index_col=0)
    y = pd.read_csv(path_labels,index_col=0)
    indexes = pd.read_csv(path_indexes)
    n_folds = 10
    train, test = iterator_cross_validation(indexes,
                                                    x,
                                                    y)
    dataset = sys.argv[2].split("/")[-1].replace(".csv","_")

    for fold in range(n_folds):

        x_train = train[fold][0]
        y_train = train[fold][1]
#        print(y_train)

        x_test = test[fold][0]
        y_test = test[fold][1]
        model = DeepForest(model_name,
                    binary_mode = binary_mode,
                    concatenate_kmers_with_svd = concatenate_kmers_with_svd,
                    transductive_mode = transductive_mode,
                    indexes_to_mask = indexes_to_mask,
                    disease_similarities = disease_similarities)
        model.fit(x_train, y_train)
        y_test_pred = np.array(model.predict_proba(x_test, y_test)) ### n_labels, n_instances, 2    
#        y_test_pred = np.array(model.predict_proba(x_test.values))[:,:,:-1].T ### n_labels, n_instances, 2    
        mode = model.get_output_name()
        print(mode)

        y_test_pred = np.squeeze(y_test_pred)
        pd.DataFrame(y_test_pred, columns = y_train.columns).to_csv(dataset + str(model_name) + "_" +  mode  + "_predictions_test_fold" + str(fold + 1) + ".csv", index=False) 
        fold +=1

    
