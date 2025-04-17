import numpy as np
import sys
import pandas as pd
sys.path.append("../../")
from deepforest import DeepForest

if __name__ == "__main__":

    path_features = sys.argv[1]
    path_labels = sys.argv[2]
    path_indexes = sys.argv[3]
    binary_mode = sys.argv[4] == "true"
    concatenate_kmers_with_svd = sys.argv[5] == "true"
    transductive_mode = sys.argv[6] == "true"
    model_index = int(sys.argv[7])
    ### RUNNING MODELS 2, 3, 9
    
    models = ['lcforest', #0
            'gcforest', #1 # Transductive
            'cafe', #2 # Transductive
            'cafe_os', #3  # Transductive
            'cafe_slc', #4
            'flaforest', #5
            'cafe_fla', #6
            'cale', #7  # Transductive
            'cale_os', #8 # Transductive
            'final_estimator' #9 # Transductive
             ]
    model_name = models[model_index]

    x = pd.read_csv(path_features, index_col=0)
    y = pd.read_csv(path_labels,index_col=0)

    dataset = sys.argv[2].split("/")[-1].replace(".csv","_")
    n_folds = 10
        
    for fold in range(n_folds):
        print(fold)
        indexes_to_mask = pd.read_csv(path_indexes.replace("fold0", "fold" + str(fold)), index_col=0)
        model = DeepForest(model_name,
                    binary_mode = binary_mode,
                    concatenate_kmers_with_svd = concatenate_kmers_with_svd,
                    transductive_mode = transductive_mode,
                    indexes_to_mask = indexes_to_mask)
        #        fake_labels = np.full(y.shape,1)
#        masked_y = np.hstack((masked_y, fake_labels))

    #    print(y)
        ## DUPLICATE FAKE COLUMN OF RESULTS
        model.fit(x, y)
        mode = model.get_output_name()
        print(mode)

        y_test_pred = np.array(model.predict_proba(x.values,y)) ### n_labels, n_instances, 2    
        pd.DataFrame(y_test_pred).to_csv(dataset + str(model_name) + "_" +  mode + "_predictions_test_fold" + str(fold + 1) + ".csv", index=False) 
        
    