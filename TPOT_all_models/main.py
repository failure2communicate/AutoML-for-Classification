import pandas as pd 

from aiflib.model import Model, _UNTRAINED_HELP
from aiflib.logger import UiPathUsageException

class Main(object):
    def __init__(self):
        self.model = Model()
        if not self.model.is_trained():
            raise UiPathUsageException(_UNTRAINED_HELP)

    def predict(self, mlskill_input):
        return self.model.predict(mlskill_input)


if __name__ == '__main__':

    main = Main()
    df = pd.read_csv('dataset\\50k_train.csv', header=0).head(20)
    df = df.drop('click', axis=1)
    json_df = df.to_json(orient='records')
    print(json_df)
    print(main.predict(json_df))