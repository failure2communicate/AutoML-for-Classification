import os
from contextlib import contextmanager
from time import time

from aiflib.model import Model

class Main:
    def __init__(self):
        self.model = Model()

    def train(self, training_directory):
        self.model.train(training_directory)

    def evaluate(self, evaluation_directory):
        return self.model.evaluate(evaluation_directory)

    def save(self):
        pass

    def process_data(self, directory):
        self.model.process_data(directory)

if __name__ == "__main__":

    # os.environ["csv_name"] = "train.csv"
    os.environ["target_column"] = "Churn"
    os.environ["max_time_mins"] = "2"        
    os.environ["warm_start"] = "false"
 
@contextmanager
def timing(description: str) -> None:
    print("-"*50)
    print("Running :", description)
    start = time()
    yield
    ellapsed_time = time() - start
    print(f"{description} time: {ellapsed_time}")
    
with timing("Initialize"):
    m = Main()
with timing("Process data"):
    m.process_data("dataset")
with timing("Evaluate"):
    m.evaluate("dataset/test")
with timing("Train"):
    m.train("dataset/training")
with timing("Evaluate"):
    m.evaluate("dataset/test")