### sets the default parameters for each run
### datasets, output path and name are added later
## m
import mlflow
import globals
import copy
import globals 
from models.estimators import estimators_dict
from sklearn.base import clone
from utils.utils import load_datasets_inductive
from utils.utils import load_datasets_transductive
from utils.utils import concatenate_datasets
from utils.utils import adjust_label_space_inductive
from utils.utils import adjust_label_space_transductive
from utils.utils import filter_label_space
import pandas as pd
import numpy as np

from evaluation.evaluate import Evaluate
import multiprocessing as mp

class Run:
        def __init__(self,
                 name,
                 train,
                 test,
                 impute,
                 task):
            self._name = name
            self._train = train
            self._test = test
            self._impute = impute
            self._task = task
            self._output_predictions = None
            self._output_performance =  None
            
        @property
        def name(self):
            return self._name
        @name.setter
        def name(self,
                 name):
            self._name =  name

        @property
        def train(self):
            return self._train
        @train.setter
        def train(self,
                train):
            self._train =  train

        @property
        def test(self):
            return self._test
        @test.setter
        def test(self,
                test):
            self._test =  test
        
        @property
        def impute(self):
            return self._impute
        @impute.setter
        def impute(self,
                impute):
            self._impute =  impute
        

        @property
        def task(self):
            return self._task
        @task.setter
        def task(self,
                task):
            self._task =  task 
        
        @property
        def output_predictions(self):
            return self._output_predictions
        @output_predictions.setter
        def output_predictions(self,
                output_predictions):
            self._output_predictions =  output_predictions
        
        @property
        def output_performance(self):
            return self._output_performance
        @output_performance.setter
        def output_performance(self,
                output_predictions):
            self._output_performance =  output_predictions
# run {
#     "name:" ## descriptive name of the run including task - year_dataset_train - year_dataset_test, e.g. INDUCTIVE_2007_2007 gives us an inductive run using train data from 2007 and test from 2007
#     "train": ## path to all folders containing the datasets used for training. The folder structure is the same as made available in the reo
#     "test": ## path to all folders containing the datasets used for test. The folder structure is the same as made available in the reo
#     "impute": ## boolean parameter regarding the imputation in the output space.
#     "task": ## transductive or inductive. Inductive will build the model using train and validation and report the performance on test. 
#             ## Transductive will build the model using train and report the performance on all three subsets.
# }

# All run prototypes are available below. The path to the CSV file used for train and test of the dataset in question are added in main.py. 
# Further, a variable "valid" regarding the path to the validation subset is also added later on in main.py

INDUCTIVE_2007_2007 = Run(
                        name = "INDUCTIVE_2007_2007",
                        train = globals.PATH_2007, 
                       test = globals.PATH_2007,
                       impute =  False,
                       task = "inductive"
                    )

INDUCTIVE_2007_2018 = Run(
                        name = "INDUCTIVE_2007_2018",
                        train = globals.PATH_2007, 
                       test = globals.PATH_2018,
                       impute = False,
                       task =  "inductive"
                    )
INDUCTIVE_2018_2018 = Run(
                        name = "INDUCTIVE_2018_2018",
                       train =  globals.PATH_2018, 
                       test =  globals.PATH_2007,
                       impute =  False,
                       task =  "inductive"
                    )

INDUCTIVE_IMPUTATION_2018_2018 = Run(
                        name = "INDUCTIVE_IMPUTATION_2018_2018",
                        train = globals.PATH_2018, 
                        test = globals.PATH_2018,
                        impute =  True,
                        task =  "inductive"
                        )

TRANSDUCTIVE_IMPUTATION_2007_2018 = Run(
                        name =  "TRANSDUCTIVE_IMPUTATION_2007_2018",
                        train =  globals.PATH_2007, 
                       test =  globals.PATH_2018,
                       impute =  True,
                       task =  "transductive"
                        )

RUN_PROTOTYPES = [
           (INDUCTIVE_2007_2007.name, INDUCTIVE_2007_2007),
           (INDUCTIVE_2007_2018.name, INDUCTIVE_2007_2018),
           (INDUCTIVE_2018_2018.name, INDUCTIVE_2018_2018),
           (TRANSDUCTIVE_IMPUTATION_2007_2018.name, TRANSDUCTIVE_IMPUTATION_2007_2018),
]


def _create_runs(run):
    new_runs = [_add_info_run(copy.deepcopy(run), dataset) for dataset in globals.DATASETS] 
    return new_runs
def _add_info_run(run, dataset):
    year_train_valid = run.train.split("/")[-1]
    year_test = run.test.split("/")[-1]
    run.name = run.name + "_" + dataset                                                         
    if run.task == "inductive":
        run.valid = run.train + "/" + dataset  + year_train_valid + "/" +  dataset + ".valid.csv"
        run.train = run.train + "/" + dataset  + year_train_valid + "/" +  dataset + ".train.csv"
        run.test = run.test + "/" + dataset + year_test +  "/" +  dataset + ".test.csv"
        new_runs = _add_models_run(run)
    elif run.task == "transductive":
        #run.valid = run.train + "/" + dataset  + year_train_valid + "/" +  dataset + ".valid.csv"
        run.train = (run.train + "/" + dataset  + year_train_valid + "/" +  dataset + ".train.csv",
                    run.train + "/" + dataset  + year_train_valid + "/" +  dataset + ".valid.csv",
                    run.train + "/" + dataset  + year_train_valid + "/" +  dataset + ".test.csv"
                    )
        run.test = (run.test + "/" + dataset  + year_test + "/" +  dataset + ".train.csv",
                    run.test + "/" + dataset  + year_test + "/" +  dataset + ".valid.csv",
                    run.test + "/" + dataset  + year_test + "/" +  dataset + ".test.csv"
                    )
#        print(run.test)
#        print(run.train)
        new_runs = _add_models_run(run)
    else:
        raise ValueError
    return new_runs
def _add_models_run(run):
    new_runs = []
    for model in globals.MODELS:
       new_run = copy.deepcopy(run)
       new_run.name = run.name + "_" + model
       new_run.output_predictions = new_run.name + "_predictions.csv"
       new_run.output_performance = new_run.name + "_results.csv" 
       new_run.model = model 
       new_runs.append(new_run)
    return new_runs


def run_experiments(run_protypes):
#    print(run_protypes)
    for setup, parameters in run_protypes:
        all_runs_datasets = _create_runs(parameters)
        with mlflow.start_run() as parent_run:
            for datasets in all_runs_datasets:
#                print(run["name"])
                for run in datasets:
                    mlflow.set_experiment(setup)
                    with mp.Pool(processes=4) as pool:
                        print("Experiment:" + run.name)
                        pool.map(_run_single_experiment(run), [])
                
def _evaluate_hierarchical(predictions, 
                          test_y, 
                          run):
        e = Evaluate()
        AUROC, AUPRC, AUPRC_w, pooled = e.CLUS_multiclass_classification_measures(
                                        predictions, 
                                        test_y, 
                                        )
        performance = pd.Series([AUROC, AUPRC, AUPRC_w, pooled], index = ["AUROC", "AUPRC", "AUPRC_w", "Pooled"])
        performance.to_csv(run.output_performance)
        mlflow.log_metric("AUROC", AUROC)
        mlflow.log_metric("AUPRC", AUPRC)
        mlflow.log_metric("AUPRC_w", AUPRC_w)
        mlflow.log_metric("Pooled", pooled)
        mlflow.log_artifact(run.output_performance)

def _run_single_experiment(run:Run):
    with mlflow.start_run(nested = True) as child_run:
        mlflow.set_tag('mlflow.runName', run.name) 
        mlflow.log_param("model", run.model)
        mlflow.log_param("task", run.task)
        mlflow.log_param("impute", run.impute)

        model = clone(estimators_dict[run.model])

        if run.task == "inductive":
            train_x, train_y = load_datasets_inductive(run.train)
            valid_x, valid_y = load_datasets_inductive(run.valid)
            test_x, test_y = load_datasets_inductive(run.test)

            train_x = concatenate_datasets(train_x, valid_x)
            train_y = concatenate_datasets(train_y, valid_y)
            train_y, test_y = adjust_label_space_inductive(train_y, test_y)

        else: ### mode == transductive

            train_x, train_y = load_datasets_transductive(run.train)
            test_x, test_y = load_datasets_transductive(run.test)
            train_y, test_y = adjust_label_space_transductive(train_y, test_y)

        train_y, test_y = filter_label_space(train_y, test_y)
        train_x.replace(np.nan_to_num(train_x.astype(np.float32))) ## workaround ValueError: Input X contains infinity or a value too large for dtype('float32').
        test_x.replace(np.nan_to_num(test_x.astype(np.float32))) ## workaround ValueError: Input X contains infinity or a value too large for dtype('float32').
        if run.model == "final_estimator":
            _single_experiment(model,
                            train_x,
                            train_y,
                            test_x,
                            test_y,
                            run)
        elif run.model in ("cafe", "cafe_os", "cale", "cale_os", "gcforest"):
            _single_deep_experiment_no_imputation(model,
                            train_x,
                            train_y,
                            test_x,
                            test_y,
                            run)
        elif run.model in ("cafe_slc", "cafe_fla", "lcforest", "flaforest"):
         _single_deep_experiment_with_imputation(model,
                            train_x,
                            train_y,
                            test_x,
                            test_y,
                            run)        
        print("Done:" + run.name)
def _single_experiment(model, 
                        train_x, 
                        train_y, 
                        test_x, 
                        test_y,
                        run):
        model.fit(train_x,
                train_y.values)
        
        predictions = model.predict_proba(test_x, hmc = True)
        _evaluate_hierarchical(predictions, test_y, run)
        _output_predictions(predictions, train_y.columns, run.output_predictions)
        mlflow.log_artifact(run.output_predictions)
        if run.task == "inductive":
            dataset_train_log = mlflow.data.from_pandas(df = train_x, name = run.train)
            dataset_test_log = mlflow.data.from_pandas(df = test_x, name = run.test)
        else:
            dataset_train_log = mlflow.data.from_pandas(df = train_x, name = run.train[0] + "_transductive")
            dataset_test_log = mlflow.data.from_pandas(df = test_x, name = run.test[0] + "_transductive")    
        mlflow.log_input(dataset = dataset_train_log, context = "training")
        mlflow.log_input(dataset = dataset_test_log, context = "test")

def _single_deep_experiment_no_imputation(model, 
                        train_x, 
                        train_y, 
                        test_x, 
                        test_y,
                        run):
        for i in range(1, 16):
            _set_run_layer_fold(run,
                                str(i)
                                ) #### update run output predictons file name and run output performance
            print("Layer:" + str(i))
            print("Experiment:" + run.name)
            _increase_max_levels(model)
            _single_experiment(model, 
                        train_x, 
                        train_y, 
                        test_x, 
                        test_y,
                        run)
            _output_Xt(model, 
                        run.name + "_Xt.csv")

            print("Done:" + run.name)

def _single_deep_experiment_with_imputation(model, 
                        train_x, 
                        train_y, 
                        test_x, 
                        test_y,
                        run):
        for i in range(1,16):
            print(i)
            _set_run_layer_fold(run,
                                str(i)
                                ) #### update run output predictons file name and run output performance
            _increase_max_levels(model)            
            print("Experiment:" + run.name)
            _single_experiment(model, 
                        train_x, 
                        train_y, 
                        test_x, 
                        test_y,
                        run)
            _output_imputed_y(model, 
                            train_y.columns, 
                            run.name + "_imputed_y.csv")
            _output_Xt(model, 
                            run.name + "_Xt.csv")

            print("Done:" + run.name)

def _set_run_layer_fold(run: Run, layer_number: str):
    
    if layer_number == "1":
        run.name = run.name + "_layer_" + layer_number 
        run.output_predictions = run.output_predictions.replace("_predictions.csv", "_layer_" + layer_number + "_predictions.csv")
        run.output_performance = run.output_performance.replace("_results.csv", "_layer_" + layer_number + "_results.csv") 
    else:
#        previous_layer_number = run.output_predictions[run.output_predictions.find("_predictions.csv") - 1]
        previous_layer_number = run.output_performance[run.output_performance.find("_results.csv") - 1]
        run.name = run.name.replace("_layer_" + previous_layer_number, "_layer_" + layer_number) 
        run.output_predictions = run.output_predictions.replace("_layer_" + previous_layer_number + "_predictions.csv", "_layer_" + layer_number + "_predictions.csv")
        run.output_performance = run.output_performance.replace("_layer_" + previous_layer_number + "_results.csv", "_layer_" + layer_number + "_results.csv") 
def _increase_max_levels(model):
    model.max_levels +=1
    print("number of levels " + str(model.max_levels))

def _output_imputed_y(model,
                    columns,
                    output_file_name: str):
    pd.DataFrame(model.last_y_resampled_, columns = columns).to_csv(output_file_name,index=False)

def _output_Xt(model,
                    output_file_name: str):
    pd.DataFrame(model.last_x).to_csv(output_file_name,index=False)


def _output_predictions(predictions: np.array,
                        columns,
                        output_file_name: str):
    predictions = pd.DataFrame(predictions, columns = columns) 
    predictions.to_csv(output_file_name, index=False)
