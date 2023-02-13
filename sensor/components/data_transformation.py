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
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import LabelEncoder

class DataTransformation:

    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,
                    data_ingestion_artifact:artifact_entity.DataIngestionArtiFact):
        try:
            self.data_transformation_config=data_transformation_config
            self.data_ingestion_artifact=data_ingestion_artifact
        except Exception as e:
            raise SensorException(e, sys)


    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            simple_imputer=SimpleImputer(strategy='constant', fill_value=0)
            robust_scaler=RobustScaler()
            pipeline = Pipeline(steps=[('Imputer', simple_imputer),
            ('RobustScaler', robust_scaler)])

            return pipeline

        except Exception as e:
            raise SensorException(e,sys)

    def initiate_data_transformation(self,)->artifact_entity.DataTransformationArtiFact:
        try:
    #reading training and testing file
            logging.info(f"reading training and testing file")
            train_df=pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df=pd.read_csv(self.data_ingestion_artifact.test_file_path)        
            logging.info(f"checking for columns -data transformation- train and test {test_df.columns} {train_df.columns}")
    #selecting input feature for train and test dataframe 
            logging.info(f"selecting input feature for train and test dataframe")      
            input_feature_train=train_df.drop(config_entity.TARGET_COLUMN,axis=1)
            input_feature_test=test_df.drop(config_entity.TARGET_COLUMN,axis=1)
            logging.info(f"checking columns input feature for train and test dataframe{list(input_feature_train.columns)}{list(input_feature_test)}")
    # selecting the target column
            logging.info(f"selecting the target column")   
            target_feature_train_df=train_df[config_entity.TARGET_COLUMN]
            target_feature_test_df=test_df[config_entity.TARGET_COLUMN]

    # encoding the target column 
            logging.info(f"encoding the target column") 
            label_encoder=LabelEncoder()
            label_encoder.fit(target_feature_train_df)

            #transformtion on target column
            logging.info(f"transformtion on target column") 
            target_feature_train_arr=label_encoder.transform(target_feature_train_df)
            target_feature_test_arr=label_encoder.transform(target_feature_test_df)
            logging.info(f"checking for columns input feature train {list(input_feature_train.columns)}")
            for i in list(input_feature_train.columns):
                logging.info(f" columns :{i}")
            transformation_pipeline=DataTransformation.get_data_transformer_object()
            transformation_pipeline.fit(input_feature_train)
            
            #tranforming input features
            logging.info(f"transforming input features")
            input_feature_train_arr=transformation_pipeline.transform(input_feature_train)
            input_feature_test_arr=transformation_pipeline.transform(input_feature_test)

            # 
            smt=SMOTETomek(sampling_strategy="minority")
            
            logging.info(f"Before resampling input training set input: {input_feature_train_arr.shape}, Target:{target_feature_train_arr.shape}")
            logging.info(f"Before resampling input test set input: {input_feature_test_arr.shape}, Target:{target_feature_test_arr.shape}")
            
            input_feature_train_arr,target_feature_train_arr=smt.fit_resample(input_feature_train_arr,target_feature_train_arr)
            input_feature_test_arr,target_feature_test_arr=smt.fit_resample(input_feature_test_arr,target_feature_test_arr)

            logging.info(f"After resampling input training set input: {input_feature_train_arr.shape}, Target:{target_feature_train_arr.shape}")
            logging.info(f"After resampling input test set input: {input_feature_test_arr.shape}, Target:{target_feature_test_arr.shape}")
                
            #target encoder
            train_arr=np.c_[input_feature_train_arr,target_feature_train_arr]
            test_arr=np.c_[input_feature_test_arr,target_feature_test_arr]
            logging.info(f"checking for columns {train_arr.shape}")
            logging.info(f"checking for columns {test_arr.shape}")

            # save numpy array
            logging.info("saving the numpy arrays for transformed test and train data")
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path, array=train_arr)
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path, array=test_arr)
    
            logging.info(f"saving the object for transformed test and train data")
        
            utils.save_object(file_path=self.data_transformation_config.tranform_object_path, obj=transformation_pipeline)
            utils.save_object(file_path=self.data_transformation_config.target_column_enoding,obj=label_encoder)        
            
            data_transformation_artifact=artifact_entity.DataTransformationArtiFact(tranform_object_path=self.data_transformation_config.tranform_object_path,transformed_train_path=self.data_transformation_config.transformed_train_path,
            transformed_test_path=self.data_transformation_config.transformed_test_path,target_encoder_path=self.data_transformation_config.target_column_enoding)

            logging.info(f"Data transformation object  {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise SensorException(e, sys)
