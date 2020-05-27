import sys
sys.path.append('..')
import time
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from tpot import TPOTClassifier
from training_only import config

class Main(object):
    def __init__(self):
        self.cur_dir = os.path.dirname(os.path.realpath(__file__))
        self.target_column = os.environ.get('target_column', 'click')
        self.feature_columns = None
        self.artifacts_directory = os.environ.get('artifacts_directory', os.path.join(self.cur_dir, 'artifacts'))
        self.train_time = os.environ.get('train_time', 2)
        self.scoring = os.environ.get('scoring', 'accuracy')
        self.keep_training = os.environ.get('keep_training', 'False')
        self.config = config.classifier_config_dict
        self.label_encoder = None
        self.model = None

        if os.path.isfile(os.path.join(self.cur_dir, 'model.sav')):
            self.model = joblib.load(os.path.join(self.cur_dir, 'model.sav'))    

        if os.path.isfile(os.path.join(self.cur_dir, 'label_encoder.sav')):
            self.label_encoder = joblib.load(os.path.join(self.cur_dir, 'label_encoder.sav'))
        else:
            self.label_encoder = LabelEncoder()

    def train(self, training_directory):
        if os.path.isfile(os.path.join(self.cur_dir, training_directory, 'train.csv')):
            data = pd.read_csv(os.path.join(self.cur_dir, training_directory, 'train.csv'), header=0, encoding="utf-8")
            self.feature_columns = data.drop(self.target_column, axis=1).columns
            (X, y) = data[self.feature_columns].values, data[self.target_column].values
        else:
            (X, y) = self.load_data(training_directory)

        if self.model is None or self.keep_training == 'True':
            self.model = self.build_model(X, y, self.artifacts_directory)
        else:
            self.model.fit(X, y)

    def build_model(self, X, y, artifacts_directory):
        # Perform missing value imputation as scikit-learn models can't handle NaN's

        pipeline_optimizer = TPOTClassifier(max_time_mins = self.train_time, 
                                            population_size=20,
                                            cv=5,
                                            scoring=self.scoring,
                                            n_jobs=-1,
                                            memory=artifacts_directory,
                                            warm_start=True,
                                            random_state=420,
                                            verbosity=2,
                                            config_dict=self.config)
        pipeline_optimizer.fit(X, y)
        
        # export fitted pipeline to artifacts directory
        pipeline_optimizer.export(os.path.join(artifacts_directory, 'TPOT_pipeline.py'))

        return pipeline_optimizer.fitted_pipeline_

    def evaluate(self, evaluation_directory):
        if os.path.isfile(os.path.join(self.cur_dir, evaluation_directory, 'evaluate.csv')):
            data = pd.read_csv(os.path.join(self.cur_dir, evaluation_directory, 'evaluate.csv'), header=0, encoding="utf-8")
            self.feature_columns = data.drop(self.target_column, axis=1).columns
            (X, y) = data[self.feature_columns].values, data[self.target_column].values
        else:
            (X, y) = self.load_data(evaluation_directory)
        return self.model.score(X, y)

    def save(self):
        joblib.dump(self.model, os.path.join(self.cur_dir, 'model.sav'))

    def load_data(self, data_directory):
        df_list = []
        for filename in os.listdir(os.path.join(self.cur_dir, data_directory)):
            if not os.path.isfile(os.path.join(self.cur_dir, data_directory, filename)):
                continue
            if filename[-4:] == '.csv':
                df_list.append(pd.read_csv(os.path.join(self.cur_dir, data_directory, filename), header=0, encoding="utf-8"))                
            
        data = pd.concat(df_list, axis=0)
        self.feature_columns = data.drop(self.target_column, axis=1).columns

        X = data[self.feature_columns].values
        y = data[self.target_column].values
        
        # Perform label encoding if labels haven't been encoded
        if data[self.target_column].dtype == 'object':
            y = self.label_encoder.fit_transform(y)
            joblib.dump(self.label_encoder, os.path.join(self.cur_dir, 'label_encoder.sav'))
        
        return X, y

    def process_data(self, data_directory):
        X, y = self.load_data(data_directory)

        # Split into train and evaluation set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=420, stratify=y)
        train_df = pd.DataFrame(X_train, columns=self.feature_columns)
        train_df[self.target_column] = y_train
        train_df.to_csv(os.path.join(self.cur_dir, data_directory, 'training', 'train.csv'), index=False)

        test_df = pd.DataFrame(X_test, columns = self.feature_columns)
        test_df[self.target_column] = y_test
        test_df.to_csv(os.path.join(self.cur_dir, data_directory, 'test', 'evaluate.csv'), index=False)

# if __name__ == "__main__":
#     t = time.time()
#     print('####### initialize ########')
#     m = Main()
#     t1 = time.time()
# #    m.input_column = 'email'
#     print('####### process data ########')
#     m.process_data('data')
#     t2 = time.time()
#     print('####### train ########')
#     m.train('data/training')
#     t3 = time.time()
#     print('####### save model ########')
#     m.save()
#     t4 = time.time()
#     print('####### evaluate ########')
#     m.evaluate('data/test')
#     t5 = time.time()
#     print('####### done ########')

#     print("initialize time: " + str(t1-t))
#     print("process data time: " + str((t2-t1)))
#     print("train time: " + str((t3-t2)))
#     print("save time: " + str((t4-t3)))
#     print("evaluate time: " + str((t5-t4)))
#     print("total time: " + str((t5-t)))
