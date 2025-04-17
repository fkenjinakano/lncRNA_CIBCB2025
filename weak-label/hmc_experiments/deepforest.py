from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    RandomForestRegressor,
    ExtraTreesRegressor,
)
import scipy.sparse
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    clone,
)
import models.label_complement
from models.cascade_learn.tree_embedder import ForestEmbedder
from models.cascade_learn.cascade import Cascade, AlternatingLevel, SequentialLevel
from models.cascade_learn import weak_labels
from models.cascade_learn.estimator_adapters import (
    RegressorAsBinaryClassifier,
    EstimatorAsTransformer,
    MultiOutputVotingRegressor,
    UnanimityClassifier,
)
from utils.transductive_converter import Converter
import copy
import numpy as np
import pandas as pd
class DeepForest:

    '''estimators_dict = {
        "lcforest": lcforest, 0 # Transductive
        "gcforest": gcforest, 1 # Transductive
        "cafe": cafe, 2  # Transductive
        "cafe_os": cafe_os, 3
        "cafe_slc": cafe_slc, 4
        "flaforest": flaforest, 5
        "cafe_fla": cafe_fla, 6
        "cale": cale,  #7 Transductive
        "cale_os": cale_os, #8 Transductive
        "final_estimator": final_estimator, 9 # Transductive
    }'''

    def __init__(self,
                model_name,
                binary_mode = False,
                concatenate_kmers_with_svd = False,
                transductive_mode = False,
                indexes_to_mask = None,
                disease_similarities = None                
                ):
        self.model_name = model_name
        self._build_estimators_dict()
        self.model = self._get_model(model_name)
        print(self.model)
        self.converter = Converter(binary_mode = binary_mode, 
                                   concatenate_kmers_with_svd = concatenate_kmers_with_svd,
                                   transductive_mode = transductive_mode, 
                                   indexes_to_mask = indexes_to_mask,
                                   disease_similarities = disease_similarities)
    def _get_model(self,
                    model_name):
        model = copy.deepcopy(self.estimators_dict[model_name])
        return model
    def fit(self,
            x,
            y):
        x, y = self.converter.process_datasets_fit(x,y)
        y = self.converter.apply_masking(y)
#        x = x.iloc[:10]
#        y = y.iloc[:10]
#        print(y)
        if self.converter.binary_mode:
            fake_labels = np.full(y.shape,1)
            y = np.hstack((y, fake_labels)) ## work around to use these models
        if type(y) is pd.DataFrame:
            y = y.values
        self.model.fit(x.values,y)
    def predict_proba(self,
                x,
                y=None):
        x = self.converter.process_datasets_predict(x,y)
#        x = x.iloc[:10]
        predictions = np.array(self.model.predict_proba(x))[:,:,1:2].T
        predictions = np.squeeze(predictions)
        if self.converter.binary_mode:
            predictions = predictions[:,0:1] 
            if not self.converter.transductive_mode:
                predictions = self.converter._unflatten_label_space(pd.DataFrame(predictions),y)
        return predictions
    def get_output_name(self):
        return self.converter.get_output_name()

    def _build_estimators_dict(self):
        RSTATE = 0  # check_random_state(0)
        #NJOBS = (cpu_count() // 5) or 1  # 1/5 of the available cores, for each CV fold
        # MEMORY = joblib.Memory(location="cache", verbose=10)
        MEMORY = None
        # NOTE: the paper undersamples for the whole forest, we perform undersampling
        # for each tree (NOW FIXED).
        MAX_EMBEDDING_SAMPLES = 1.0
        # Maximum fraction of samples in a tree node for it to be used in the embeddings
        MAX_NODE_SIZE = 0.95
        N_COMPONENTS = 0.8
        # N_COMPONENTS = "mle"  # Use Minka's (2000) MLE to determine the number of components
        VERBOSE = 0
        FOREST_PARAMS = dict(
            n_estimators=150,
#            n_estimators=1,
            max_samples=None,
            max_features="sqrt",
            min_samples_leaf=5,
            max_depth=None,
            n_jobs=-1,
            verbose=True,
            random_state=RSTATE,
        )
        CASCADE_PARAMS = dict(
        #    max_levels=1,
            max_levels=3,
            verbose=VERBOSE,
            memory=MEMORY,
            keep_original_features=True,
        )


        class Densifier(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self

            def transform(self, X):
                if scipy.sparse.issparse(X):
                    return X.toarray()
                return X


        rf_embedder = (
            ForestEmbedder(
                RandomForestRegressor(
                    **FOREST_PARAMS,
                    #max_samples=MAX_EMBEDDING_SAMPLES,
                    bootstrap=True,  # Default for RF
                ),
                method="path",
                node_weights="log_node_size",  # Eq. (1)
                max_node_size=MAX_NODE_SIZE,
            )
        )

        xt_embedder = (
            ForestEmbedder(
                ExtraTreesRegressor(
                    **FOREST_PARAMS,
                    #max_samples=MAX_EMBEDDING_SAMPLES,
                    bootstrap=True,
                ),
                method="path",
                node_weights="log_node_size",  # Eq. (1)
                max_node_size=0.8,
            )
        )

        xt_embedder_pca = Pipeline([
            ("embedder", xt_embedder),
            ("densifier", Densifier()),
            ("pca", PCA(n_components=N_COMPONENTS, random_state=RSTATE)),
        ])

        rf_embedder_pca = Pipeline([
            ("embedder", rf_embedder),
            ("densifier", Densifier()),
            ("pca", PCA(n_components=N_COMPONENTS, random_state=RSTATE)),
        ])

        xt_proba_transformer = EstimatorAsTransformer(
            ExtraTreesRegressor(
                **FOREST_PARAMS,
                # oob_score=True,  # Only necessary for final estimator
                #max_samples=None,
                bootstrap=True,  # Default for RF
            ),
        )
        rf_proba_transformer = EstimatorAsTransformer(
            RandomForestRegressor(
                **FOREST_PARAMS,
                # oob_score=True,  # Only necessary for final estimator
                # max_samples=None,
                bootstrap=True,
            ),
        )

        final_estimator = RegressorAsBinaryClassifier(
            MultiOutputVotingRegressor(
                estimators=[
                    (
                        "rf",
                        RandomForestRegressor(
                            **FOREST_PARAMS,
                            oob_score=True,
                            #max_samples=None,
                            bootstrap=True,  # Default for RF
                        ),
                    ),
                    (
                        "xt",
                        ExtraTreesRegressor(
                            **FOREST_PARAMS,
                            oob_score=True,
                            #max_samples=None,
                            bootstrap=True,
                        ),
                    ),
                ],
            )
        )


        # Copies final_estimator.
        fixed_level_proba = FeatureUnion([
            ("rf", rf_proba_transformer),
            ("xt", xt_proba_transformer),
        ])

        alternating_level_embedding = AlternatingLevel([
            ("rf", rf_embedder_pca),
            ("xt", xt_embedder_pca),
        ])

        alternating_level_proba = AlternatingLevel([
            ("xt", xt_proba_transformer),
            ("rf", rf_proba_transformer),
        ])

        alternating_level_embedding_proba = AlternatingLevel([
            ("rf", FeatureUnion([("embedder", rf_embedder_pca), ("proba", rf_proba_transformer)])),
            ("xt", FeatureUnion([("embedder", xt_embedder_pca), ("proba", xt_proba_transformer)])),
        ])


        imputer_estimator = MultiOutputVotingRegressor(
            estimators=[
                (
                    "rf",
                    RandomForestRegressor(
                        **FOREST_PARAMS,
                        oob_score=True,
                        #max_samples=None,
                        bootstrap=True,  # Default for RF
                    ),
                ),
                (
                    "xt",
                    ExtraTreesRegressor(
                        **FOREST_PARAMS,
                        oob_score=True,
                        #max_samples=None,
                        bootstrap=True,
                    ),
                ),
            ],
        )

        scar_imputer = weak_labels.SCARImputer(
            label_freq_percentile=0.95,
            verbose=True,
            estimator=imputer_estimator,
        )

        lc_imputer = weak_labels.LabelComplementImputer(
            label_freq_percentile=0.95,
            verbose=True,
            estimator=imputer_estimator,
            weight_proba=False,
        )

        wang_imputer = models.label_complement.LabelComplementImputer(
            estimator=UnanimityClassifier(
                estimators=[
                    ("rf", RandomForestClassifier(**FOREST_PARAMS)),
                    ("et", ExtraTreesClassifier(**FOREST_PARAMS)),
                ],
                threshold=0.4,
            ),
            verbose=True,
            tice_params=dict(max_bepp=5, max_splits=500, min_set_size=5),  # random_state=RSTATE),
            cv_params=dict(cv=5),
        )

        # zhou_level = FeatureUnion(  # Too slow.
        #     [
        #         ("rf", EstimatorAsTransformer(CVRegressor(RandomForestRegressor(**FOREST_PARAMS), cv=5))),
        #         ("xt", EstimatorAsTransformer(CVRegressor(ExtraTreesRegressor(**FOREST_PARAMS), cv=5))),
        #     ]
        # )


        # ===========================================================================
        # Cascade estimators
        # ===========================================================================

        cafe = clone(Cascade(
            level=alternating_level_embedding,
            final_estimator=final_estimator,
            **CASCADE_PARAMS,
        ))

        cafe_os = clone(Cascade(
            level=alternating_level_embedding_proba,
            final_estimator=final_estimator,
            **CASCADE_PARAMS,
        ))

        slcforest = clone(Cascade(
            level=SequentialLevel([
                ("alternating_forests", alternating_level_proba),
                ("label_imputer", lc_imputer),
            ]),
            final_estimator=final_estimator,
            **CASCADE_PARAMS,
        ))

        cafe_slc = clone(Cascade(
            level=SequentialLevel([
                ("alternating_forests", alternating_level_embedding_proba),
                ("label_imputer", lc_imputer),
            ]),
            final_estimator=final_estimator,
            **CASCADE_PARAMS,
        ))

        flaforest = clone(Cascade(
            level=SequentialLevel([
                ("alternating_forests", alternating_level_proba),
                ("label_imputer", scar_imputer),
            ]),
            final_estimator=final_estimator,
            **CASCADE_PARAMS,
        ))

        cafe_fla = clone(Cascade(
            level=SequentialLevel([
                ("alternating_forests", alternating_level_embedding_proba),
                ("label_imputer", scar_imputer),
            ]),
            final_estimator=final_estimator,
            **CASCADE_PARAMS,
        ))

        lcforest = clone(Cascade(
            level=SequentialLevel([
                # ("transformer", zhou_level),
                ("transformer", fixed_level_proba),
                ("label_imputer", wang_imputer),
            ]),
            final_estimator=final_estimator,
            **CASCADE_PARAMS,
        ))

        gcforest = clone(Cascade(
            # level=zhou_level,
            level=fixed_level_proba,
            final_estimator=final_estimator,
            **CASCADE_PARAMS,
        ))

        self.estimators_dict = {
            "lcforest": lcforest,
            "gcforest": gcforest,
            "cafe": cafe,
            "cafe_os": cafe_os,
            "cafe_slc": cafe_slc,
            "flaforest": flaforest,
            "cafe_fla": cafe_fla,
            "final_estimator": final_estimator,
        }
