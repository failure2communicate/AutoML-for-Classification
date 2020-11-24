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