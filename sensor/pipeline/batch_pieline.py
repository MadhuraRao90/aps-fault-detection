from sensor.exception import SensorException
from sensor.logger import logging
from sensor.predictor import ModelResolver
import pandas as pd
from datetime import datetime
from sensor.utils import load_object
import os,sys
import numpy as np
PREDICTION_DIR="prediction"
#PREDICTION_FILE_NAME=f"{datetime.now().strftime("%m%d%Y__%H%M%S")}"
def batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR,exist_ok=True)
        model_resolver=ModelResolver()
        logging.info(f"Reading the file : {input_file_path}")
        prediction_file_name=os.path.basename(input_file_path).replace(".csv",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        transformer=load_object(file_path=model_resolver.get_latest_transformer_path())
        input_features=list(transformer.feature_names_in_)
        print("features",input_features)
        df=pd.read_csv(input_file_path,index_col=[0])
        print(df.columns)
        #df.drop('Unnamed:0',axis=0)
        df.replace({"na":np.NAN},inplace=True)
        # write code for validation 

        logging.info("loading transformer to transform input data")
        transformer=load_object(file_path=model_resolver.get_latest_transformer_path())
        input_features=list(transformer.feature_names_in_)
        transformed_input= transformer.transform(df[input_features])
        
        logging.info("loading model to make prediction")
        model=load_object(file_path=model_resolver.get_latest_model_path())
        y_pred=model.predict(transformed_input)
        
        logging.info("target encoder to convert predicted value into categorical column")
        target_encoder=load_object(file_path=model_resolver.get_latest_target_encoder_path())
        car_prediction=target_encoder.inverse_transform(y_pred)
        input_arr=df[input_features]
        df['Prediction']=y_pred
        df['cat_pred']=car_prediction
        prediction_file_path=os.path.join(PREDICTION_DIR,prediction_file_name)
        df.to_csv(prediction_file_path,index=False,header=True)
        return prediction_file_path
        

        
    except Exception as e:
        raise SensorException(e,sys)