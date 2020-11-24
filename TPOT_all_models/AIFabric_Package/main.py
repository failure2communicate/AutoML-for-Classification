from aiflib.model import Model, _UNTRAINED_HELP
from aiflib.logger import UiPathUsageException

class Main(object):
    def __init__(self):
        self.model = Model()
        if not self.model.is_trained():
            raise UiPathUsageException(_UNTRAINED_HELP)

    def predict(self, mlskill_input):
        return self.model.predict(mlskill_input)