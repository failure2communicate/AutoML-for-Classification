import sys
sys.path.append('..')

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from tpot import TPOTClassifier
from training_only import config

class Main(object):
    def __init__(self):
        self.cur_dir = os.path.dirname(os.path.realpath(__file__))
        self.config = config.classifier_config_dict
        self.model_path = './model.sav'
        self.model = None

        if os.path.isfile(os.path.join(self.cur_dir, self.model_path)):
            self.model = joblib.load(os.path.join(self.cur_dir, self.model_path))

    def train(self, training_directory):
        (X, y) = self.load_data(os.path.join(training_directory, 'train.csv'))
        if self.model is None:
            self.model = self.build_model(X, y)
        else:
            self.model.fit(X, y)

    def build_model(self, X, y):
        # Perform missing value imputation as scikit-learn models can't handle NaN's
        nan_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        X = nan_imputer.fit_transform(X)

        pipeline_optimizer = TPOTClassifier(generations=5,
                                            population_size=20,
                                            cv=5,
                                            n_jobs=1,
                                            warm_start=False,
                                            random_state=42,
                                            verbosity=2,
                                            config_dict=self.config)
        pipeline_optimizer.fit(X, y)

        pipe = Pipeline([('nan_imputer', nan_imputer),
                         ('tpot_pipeline', pipeline_optimizer.fitted_pipeline_)])
        return pipe

    def evaluate(self, evaluation_directory):
        (X, y) = self.load_data(os.path.join(evaluation_directory, 'evaluate.csv'))
        return self.model.score(X, y)

    def save(self):
        joblib.dump(self.model, os.path.join(self.cur_dir, self.model_path))

    def load_data(self, path):
        df = pd.read_csv(path, header=None)
        X = df.iloc[:, :-1].copy()
        y = df.iloc[:, -1].copy()
        return X, y


# if __name__ == "__main__":
#     m = Main()
#     m.train('Data')
#     m.save()
