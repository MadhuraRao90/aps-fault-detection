from sensor import utils
from sensor.entity import config_entity
from sensor.entity import artifact_entity
from sensor.logger import logging
from sensor.exception import SensorException
import sys, os
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self,data_ingestion_config:config_entity.DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            logging.debug(f"{SensorException(e, sys)}")
            raise SensorException(e, sys)
    def initiate_data_ingestion(self)->artifact_entity.DataIngestionArtiFact:
        try:
            # Exporting collection data as dataframe
            logging.info(f"Exporting collection data as dataframe")
            df:pd.DataFrame=utils.get_collection_as_dataframe(self.data_ingestion_config.database_name, self.data_ingestion_config.collection_name)
            logging.info(f"dataframe {df}")
            #replacing the NA values and saving in feature_store
                # create feature_store directory is not available
            logging.info(f"Replacing the NA values and saving in feature store")
            df.replace(to_replace="na",value=np.NAN,inplace=True)

            feature_store_dir=os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir,exist_ok=True)

            # save the df as csv
            logging.info("save the dataframe as csv")
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_file_path,index=False,header=True)
            #logging.info(f"checking for columns -data ingestion :{list(df.columns)}")
            
            #Splitting the data into train and test
            logging.info("splitting the data into train and test data ")
            train_df,test_df=train_test_split(df,test_size=self.data_ingestion_config.test_size,
            random_state=42)

            # save train and test data into respective folders

            logging.info("Save train and test data into their respective folders")
            train_store_dir=os.path.dirname(self.data_ingestion_config.train_file_path)
            test_store_dir=os.path.dirname(self.data_ingestion_config.test_file_path)
            os.makedirs(train_store_dir,exist_ok=True)
            os.makedirs(test_store_dir,exist_ok=True)
            train_df.to_csv(path_or_buf=self.data_ingestion_config.train_file_path,index=False,header=True)
            test_df.to_csv(path_or_buf=self.data_ingestion_config.test_file_path,index=False,header=True)

            # Prepare artifact 
            logging.info(f"Preparing the artifact for data ingestion")

            data_ingestion_artifact=artifact_entity.DataIngestionArtiFact(feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
            train_file_path=self.data_ingestion_config.train_file_path,test_file_path=self.data_ingestion_config.test_file_path)
            logging.info(f"data_ingestion_artifact:{data_ingestion_artifact}")
            return data_ingestion_artifact


        except Exception as e:
            logging.debug(f"{SensorException(e, sys)}")
            raise SensorException(e, sys)