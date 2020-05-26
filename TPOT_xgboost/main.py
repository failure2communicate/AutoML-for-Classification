from collections import defaultdict
import sklearn
import joblib 
import pandas as pd
import numpy as np
import json
import os

class Main(object): 
    def __init__(self): 
        """ Initializes the model and all ancilliary data (e.g. word embeddings) """
        self.cur_dir = os.path.dirname(os.path.realpath(__file__))
        self.model = joblib.load(os.path.join(self.cur_dir, 'model.sav'))
        self.label_encoder = None
        
        if os.path.isfile(os.path.join(self.cur_dir, 'label_encoder.sav')):
            self.label_encoder = joblib.load(os.path.join(self.cur_dir, 'label_encoder.sav'))
    
    def predict(self, mlskill_input): 
        """ Once an ML Package is deployed as an ML Skill, this function will 
        be the endpoint callable by outside clients. If calling this ML Skill 
        through UiPath Studio, if the ML Skill was created with input type set 
        to file, the client will send a file in serialized bytes. 
        
        An example implementation for deserializing an image file: 
        ... 
        def predict(self, mlskill_input):
            from PIL import image 
            import io 
            image = Image.open(io.BytesIO(mlskill_input)) 
            
        :param str mlskill_input: input coming from a client. """ 
        import pandas as pd
        data = pd.read_json(mlskill_input)
        data_dict = defaultdict(list)
        
        # # Not all scikit-learn models support the predict_proba function
        try:
            prediction_tuples = self.model.predict_proba(data.values)
            for prediction_tuple in prediction_tuples:
                prediction = np.argmax(prediction_tuple)
                confidence = prediction_tuple[prediction]
                data_dict['Predictions'].append(prediction)
                data_dict['Confidences'].append(confidence)
                if self.label_encoder is not None:
                    label = self.label_encoder.inverse_transform([prediction])[0]
                    data_dict['Labels'].append(label)    
            return_df = pd.DataFrame([data_dict])
            return return_df.to_json(orient = 'records')
        except:
            predictions = self.model.predict(data.values)
            data_dict['Predictions'] = predictions
            if self.label_encoder is not None:
                labels = self.label_encoder.inverse_transform(predictions)
                data_dict['Labels'] = labels          
            return_df = pd.DataFrame([data_dict])
            return return_df.to_json(orient = 'records')
            
if __name__ == '__main__':
    main = Main()
    df = pd.read_csv('Benchmark_data\\50k_train.csv', header=0).head(20)
    df = df.drop('click', axis=1)
    json_df = df.to_json(orient='records')
    print(main.predict(json_df))

