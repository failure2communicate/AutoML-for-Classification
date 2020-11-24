import logging
import os
import numpy as np
from functools import partial

class ConfigValidator:
    class __Singleton:
        def __init__(self):
            self.has_logged = False

        def __str__(self):
            return repr(self) + self.has_logged

    instance = None
    def __init__(self):
        if not ConfigValidator.instance:
            ConfigValidator.instance = ConfigValidator.__Singleton()

    def finalize(self):
        ConfigValidator.instance.has_logged = True

    def log(self, msg):
        logging.getLogger("parameters-validator").info(msg)
        if not ConfigValidator.instance.has_logged:
            logging.getLogger("parameters-validator").info(msg)

def is_folder_empty(path):
    if os.path.exists(path) and not os.path.isfile(path):   
        if not os.listdir(path): 
            return True
        else: 
            return False
    else:
        return False

def os_int(name, default, condition, message):
    if name in os.environ:
        try:
            value = int(os.environ[name])
        except Exception as e:
            ConfigValidator().log(f"Bad usage, parameter [{name}] should be parseable as an integer defaulting to value [{default}]")
            return default
        else:
            if not condition(value):
                ConfigValidator().log(f"Bad usage, parameter [{name}], {message}, defaulting to [{default}]")            
                return default
            else:
                return value
    return default

def os_float(name, default, condition, message):
    if name in os.environ:
        try:
            value = float(os.environ[name])
        except Exception as e:
            ConfigValidator().log(f"Bad usage, parameter [{name}] should be parseable as float defaulting to value [{default}]")
            return default
        else:
            if not condition(value):
                ConfigValidator().log(f"Bad usage, parameter [{name}], {message}, defaulting to value [{default}]")
                return default
            else:
                return value
    return default

def os_param(name, default, condition, message):
    if name in os.environ:
        if not condition(os.environ[name]):
            ConfigValidator().log(f"Bad usage, parameter [{name}], {message}, defaulting to [{default}]")  
            return default
        return os.environ[name]
    return os.environ.get(name, default)

def os_flag(name, default="false"):
    if name in os.environ:
        if os.environ[name].lower() not in ["true", "false"]:
            ConfigValidator().log(f"Bad usage, parameter [{name}] should be one of [true|false], defaulting to value [{default}]")
            return default == "true"
        return os.environ[name] == "true"
    return default == "true"


class Config:
    """
    Configuration driven by environment variables.
    """
    _exposed_variables = set([
        "target_column", "csv_name", "encoding", "encoding", "scoring", "max_time_mins", "warm_start",
        "generations", "population_size", "offspring_size", "mutation_rate", "crossover_rate",
        "cv", "subsample", "max_eval_time_mins", "early_stop", "random_seed", "percentage_evaluate",
    ])

    def __init__(self):
        self.cur_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        join_path = partial(os.path.join, self.cur_dir)
        unconditional = lambda x: True

        # Trace which exposed variables where set.
        self.specified_parameters = Config._exposed_variables & set(os.environ.keys())

        #####################################
        #          Data parameters          #
        #####################################
        self.target_column = os_param(
            "target_column", "target", unconditional, ""
        )
        self.csv_name = os_param(
            "csv_name", None, unconditional, ""
        )
        self.delimiter = os_param(
            "csv_delimiter", ",", unconditional, ""
        )
        self.encoding = os_param(
            "encoding", "utf-8", unconditional, ""
        )
        #####################################
        #       Basic model parameters      #
        #####################################
        permissible_metrics = [
            "accuracy", "adjusted_rand_score", "average_precision", "balanced_accuracy", 
            "f1", "f1_macro", "f1_micro", "f1_samples", "f1_weighted", "neg_log_loss", 
            "precision", "precision_macro", "precision_micro", "precision_samples", "precision__weighted",
            "recall", "recall_macro", "recall_micro", "recall_samples", "recall__weighted",
            "jaccard","jaccard_macro", "jaccard_micro", "jaccard_samples", "jaccard__weighted",
            "roc_auc", "roc_auc_ovr", "roc_auc_ovo", "roc_auc_ovr_weighted", "roc_auc_ovo_weighted"
        ]
        self.scoring = os_param(
            "scoring", "accuracy", lambda x: x in permissible_metrics,
            f"evaluation metric to be returned must be one of [{permissible_metrics}]"
        )
        self.max_time_mins = os_int(
            "max_time_mins", 2, lambda x: x > 0,
            f"maximum training time must be greater than 0"
        )
        self.warm_start = os_flag(
            "warm_start","false"
        )

        #####################################
        #     Advanced model parameters     #
        #####################################
         
         # Validates if all classes have at least some number of minium training examples
        self.class_percentage_validation = os_float(
            "class_percentage_validation", 0.01, lambda x: x>= 0 and x<1,
            "percentage used for validation should be greater than 0 and less than 1"
        )
        self.generations = os_int(
            "generations", 100, lambda x: x > 0,
            "number of generations must be greater than 0"
        )
        self.population_size = os_int(
            "population_size", 100, lambda x: x > 0,
            "number of individuals must be greater than 0"
        )
        self.offspring_size = os_int(
            "offspring_size", None, lambda x: x > 0,
            "number of offspring must be greater than 0"
        )
        self.mutation_rate = os_float(
            "mutation_rate", 0.9, lambda x: x >= 0 and x <= 1.0,
            "mutation_rate must be in the range [0.0, 1.0]"
        )
        self.crossover_rate = os_float(
            "crossover_rate", 0.1, lambda x: x >= 0 and x <= 1.0,
            "crossover_rate must be in the range [0.0, 1.0]"
        )
        self.cv = os_int(
            "cv", 5, lambda x: x >= 2 and x <= 10,
            "number of cross-validation folds must be an integer between 2 and 10"
        )
        self.subsample = os_float(
            "subsample", 1, lambda x: x >= 0 and x < 1.0,
            "fraction of training samples that are used during the TPOT optimization process must be in the range (0.0, 1.0]"
        )
        self.max_eval_time_mins = os_int(
            "max_eval_time_mins", 5, lambda x: x > 0,
            "max_eval_time_mins must be greater than 0"
        )
        self.early_stop = os_int(
            "early_stop", None, lambda x: x > 0,
            "early_stop must be greater than 0"
        )
        self.seed = os_int(
            "random_seed", 0, unconditional, "")

        #####################################
        #      Process Data Parameters      #
        #####################################

        self.process_data_split_percentage = os_float(
            "percentage_evaluate", .2,
            lambda x: x >= 0 and x < 1,
            "test percentage should be greater than or equal to 0 and less than 1"
        )
        self.train_data_directory = os_param(
            "training_data_directory",
            join_path("dataset", "training"),
            lambda x: os.path.exists(x),
            "train data directory should exist"
        )
        self.test_data_directory = os_param(
            "test_data_directory",
            join_path("dataset", "test"),
            lambda x: os.path.exists(x),
            "test data directory should exist"
        )
        self.artifacts_directory = os_param(
            "artifacts_directory",
            join_path("artifacts"),
            lambda x: os.path.exists(x),
            "path should point to an existing directory"
        )
        
        # Check if test data has been selected from the UI
        self.test_data_from_ui = not is_folder_empty(self.test_data_directory)
        
        #####################################
        #        Logging Parameters         #
        #####################################

        self.is_debug = os_flag("debug")
        self.is_verbose = os_flag("verbose") and not self.is_debug
        self.is_info = not (self.is_debug or self.is_verbose)

        # If is debug, set root logger to debug to see logs from other
        # modules (e.g. torch, fastai, etc.)
        if self.is_debug:
            logging.basicConfig(format="[%(asctime)s] [%(name)s] %(message)s", level = logging.DEBUG)

        # Else set as warning so logs from other modules not populated.
        else:
            logging.basicConfig(format="[%(asctime)s] %(message)s", level = logging.WARNING)

        #####################################
        #       TPOT Config Parameters      #
        #####################################        
        # Check the TPOT documentation for information on the structure of config dicts
        
        self.classifier_config_dict = {

            # Classifiers
            "sklearn.naive_bayes.GaussianNB": {
            },

            "sklearn.naive_bayes.BernoulliNB": {
                "alpha": [1e-3, 1e-2, 1e-1, 1., 10., 100.],
                "fit_prior": [True, False]
            },

            "sklearn.naive_bayes.MultinomialNB": {
                "alpha": [1e-3, 1e-2, 1e-1, 1., 10., 100.],
                "fit_prior": [True, False]
            },

            "sklearn.ensemble.ExtraTreesClassifier": {
                "n_estimators": [100],
                "criterion": ["gini", "entropy"],
                "max_features": np.arange(0.05, 1.01, 0.05),
                "min_samples_split": range(2, 21),
                "min_samples_leaf": range(1, 21),
                "bootstrap": [True, False]
            },

            "sklearn.ensemble.GradientBoostingClassifier": {
                "n_estimators": [100],
                "learning_rate": [1e-3, 1e-2, 1e-1, 0.5, 1.],
                "max_depth": range(1, 11),
                "min_samples_split": range(2, 21),
                "min_samples_leaf": range(1, 21),
                "subsample": np.arange(0.05, 1.01, 0.05),
                "max_features": np.arange(0.05, 1.01, 0.05)
            },

            "sklearn.neighbors.KNeighborsClassifier": {
                "n_neighbors": range(1, 101),
                "weights": ["uniform", "distance"],
                "p": [1, 2]
            },

            "sklearn.svm.LinearSVC": {
                "penalty": ["l1", "l2"],
                "loss": ["hinge", "squared_hinge"],
                "dual": [True, False],
                "tol": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                "C": [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]
            },

            "sklearn.linear_model.LogisticRegression": {
                "max_iter": [100000],
                "penalty": ["l1", "l2"],
                "C": [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
                "dual": [True, False]
            },

            "sklearn.linear_model.SGDClassifier": {
                "loss": ["log", "hinge", "modified_huber", "squared_hinge", "perceptron"],
                "penalty": ["elasticnet"],
                "alpha": [0.0, 0.01, 0.001],
                "learning_rate": ["invscaling", "constant"],
                "fit_intercept": [True, False],
                "l1_ratio": [0.25, 0.0, 1.0, 0.75, 0.5],
                "eta0": [0.1, 1.0, 0.01],
                "power_t": [0.5, 0.0, 1.0, 0.1, 100.0, 10.0, 50.0]
            },

            # Preprocesssors
            "sklearn.preprocessing.Binarizer": {
                "threshold": np.arange(0.0, 1.01, 0.05)
            },

            "sklearn.decomposition.FastICA": {
                "tol": np.arange(0.0, 1.01, 0.05)
            },

            "sklearn.cluster.FeatureAgglomeration": {
                "linkage": ["ward", "complete", "average"],
                "affinity": ["euclidean", "l1", "l2", "manhattan", "cosine"]
            },

            "sklearn.preprocessing.MaxAbsScaler": {
            },

            "sklearn.preprocessing.MinMaxScaler": {
            },

            "sklearn.preprocessing.Normalizer": {
                "norm": ["l1", "l2", "max"]
            },

            "sklearn.kernel_approximation.Nystroem": {
                "kernel": ["rbf", "cosine", "chi2", "laplacian", "polynomial", "poly", "linear", "additive_chi2", "sigmoid"],
                "gamma": np.arange(0.0, 1.01, 0.05),
                "n_components": range(1, 11)
            },

            "sklearn.decomposition.PCA": {
                "svd_solver": ["randomized"],
                "iterated_power": range(1, 11)
            },

            "sklearn.preprocessing.PolynomialFeatures": {
                "degree": [2],
                "include_bias": [False],
                "interaction_only": [False]
            },

            "sklearn.kernel_approximation.RBFSampler": {
                "gamma": np.arange(0.0, 1.01, 0.05)
            },

            "sklearn.preprocessing.RobustScaler": {
            },

            "sklearn.preprocessing.StandardScaler": {
            },

            "tpot.builtins.ZeroCount": {
            },

            "tpot.builtins.OneHotEncoder": {
                "minimum_fraction": [0.05, 0.1, 0.15, 0.2, 0.25],
                "sparse": [False],
                "threshold": [10]
            },

            # Selectors
            "sklearn.feature_selection.SelectFwe": {
                "alpha": np.arange(0, 0.05, 0.001),
                "score_func": {
                    "sklearn.feature_selection.f_classif": None
                }
            },

            "sklearn.feature_selection.SelectPercentile": {
                "percentile": range(1, 100),
                "score_func": {
                    "sklearn.feature_selection.f_classif": None
                }
            },

            "sklearn.feature_selection.VarianceThreshold": {
                "threshold": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
            },

            "sklearn.feature_selection.RFE": {
                "step": np.arange(0.05, 1.01, 0.05),
                "estimator": {
                    "sklearn.ensemble.ExtraTreesClassifier": {
                        "n_estimators": [100],
                        "criterion": ["gini", "entropy"],
                        "max_features": np.arange(0.05, 1.01, 0.05)
                    }
                }
            },

            "sklearn.feature_selection.SelectFromModel": {
                "threshold": np.arange(0, 1.01, 0.05),
                "estimator": {
                    "sklearn.ensemble.ExtraTreesClassifier": {
                        "n_estimators": [100],
                        "criterion": ["gini", "entropy"],
                        "max_features": np.arange(0.05, 1.01, 0.05)
                    }
                }
            }

        }

        # Finalize config validation.
        ConfigValidator().finalize()


# if __name__ == "__main__":
#     c = Config()
#     print(c.train_data_directory)