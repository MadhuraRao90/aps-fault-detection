from sensor.logger import logging
from sensor.exception import SensorException
from pandas import pandas
from sensor.entity import config_entity,artifact_entity
from typing import Optional
import os,sys
from scipy.stats import ks_2samp
import numpy as np
from sensor import utils
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import f1_score 
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

class ModelTrainer:
    try:
        def __init__(self,model_trainer_config:config_entity.ModelTrainerConfig,
        data_transformation_artifact:artifact_entity.DataTransformationArtiFact,
        ):
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
    except Exception as e:
        raise SensorException(e,sys)
    def fine_tune_model(self,X,y):

        try:
            param=self.model_trainer_config.param
            xgb=XGBClassifier()
            halving_grid=HalvingGridSearchCV(xgb,param)
            halving_grid.fit(X,y)
            return halving_grid            
            
        except Exception as e:
            raise SensorException(e, sys)
    def train_model(self,X,y):
        try:
            xgb_clf=XGBClassifier()
            xgb_clf.fit(X,y)
            return xgb_clf 
        except Exception as e:
            raise SensorException(e, sys)  

    def initiate_model_trainer(self,)-> artifact_entity.ModelTrainerArtiFact:
        try:
            logging.info(f"loading test and train array")
            train_arr=utils.load_numpy_array_data(self.data_transformation_artifact.transformed_train_path)            
            test_arr=utils.load_numpy_array_data(self.data_transformation_artifact.transformed_test_path)
           
            logging.info(f" seperating input features and target features from both test and train arrays")
            X_train,y_train=train_arr[:,:-1],train_arr[:,-1]
            X_test,y_test=test_arr[:,:-1],test_arr[:,-1]

            logging.info(" Train the model")
            model=self.train_model(X=X_train,y=y_train)

            #logging.info("Hypertuning the model to get the best params")       
            #model_fine_tuned=self.fine_tune_model(X=X_train,y=y_train)
            #yhat_train=model_fine_tuned.predict(X_train)
            #yhat_test=model_fine_tuned.predict(X_test)
            #logging.info(f"The best params for XGBclassifier is : {model_fine_tuned.best_params_}")
            #logging.info("Hypertuning the model to get the best params")       
            
            # comment if using hypertuning using halving search cv
            yhat_train=model.predict(X_train)
            yhat_test=model.predict(X_test)
            
            logging.info("calculating f1 train_score")
            f1_score_train= f1_score(y_train,yhat_train)
            
            logging.info("calculating f1 test_score")
            f1_score_test= f1_score(y_test,yhat_test)

            logging.info(f"f1 score train :{f1_score_train} , f1 score test :{f1_score_test}")
            logging.info("checking if the model is underfitting")        
            # check for overfitting or underfitting or expected score
            if f1_score_test < self.model_trainer_config.expected_score:
                raise Exception(f"model is not good as it is not able to give the expected accuracy :{self.model_trainer_config.expected_score} , model score :{f1_score_test}")

            logging.info("checking if the model is overfitting")
            diff= abs(f1_score_train-f1_score_test) 
            if diff > self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and test score diff :{diff} is more than the overfitting threshold :{model_trainer_config.overfitting_threshold}")
            
            logging.info("saving the model")
            utils.save_object(self.model_trainer_config.model_trainer_path, model)

            # prepare artifact
            logging.info("preparing the model trainer artifact")

            model_trainer_artifact=artifact_entity.ModelTrainerArtiFact(model_trainer=self.model_trainer_config.model_trainer_path,
            f1_test_score=f1_score_test,f1_train_score=f1_score_train)
            return model_trainer_artifact
        except Exception as e:
            raise SensorException(e, sys)
