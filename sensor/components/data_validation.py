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

class DataValidation:

    def __init__(self,
                    data_validation_config:config_entity.DataValidationConfig,
                    data_ingestion_artifact:artifact_entity.DataIngestionArtiFact):
        try:
            logging.info(f"{'>>'*20} Data Validation {'<<'*20}" )

            self.data_validation_config=data_validation_config
            self.validation_error=dict()
            self.data_ingestion_artifact=data_ingestion_artifact
            logging.info("here")
        except Exception as e:
            raise SensorException(e, sys)

    

    def drop_missing_values_columns(self,df:pd.DataFrame,report_key_name:str) -> Optional[pd.DataFrame]:
        """This function will drop columns which contains missing value more than specified threshold
        df: Accepts pandas dataframe
        threshold : Percentage criteria to drop a column
        ======================================================================================

        returns pandas Dataframe if atleast single column is available 

        """
        try:
            threshold=self.data_validation_config.missing_threshold
            # selecting column names which contains null values less than threshold 
            logging.info(f"selecting column names which contains null values above {self.data_validation_config.missing_threshold}  ")

            null_report=df.isna().sum()/df.shape[0]
            drop_column_names=null_report[null_report>threshold].index

            logging.info(f"columns to drop:{list(drop_column_names)}")
            self.validation_error[report_key_name]=list(drop_column_names)
            df.drop(list(drop_column_names),axis=1,inplace=True)
            # return none if no columns left
            logging.info("return columns after removing columns with values less than threshold")
            if len(df.columns)==0:
                return None
            return df
        except Exception as e:
            raise SensorException(e, sys)

    def is_required_columns_exist(self,base_df,current_df,report_key_name:str) ->bool:
        try:
            logging.info(f"Checking if the columns in base exist in the current dataframe for test and train")
            base_columns=base_df.columns
            current_columns=current_df.columns
            missing_columns=[]
            for base_column in base_columns:
                if base_column not in current_columns:
                    logging.info(f"Column {base_column} is not available")
                    missing_columns.append(base_column)
            
            if len(missing_columns) >0:
                self.validation_error[report_key_name] =missing_columns
                return False
            return True
        except Exception as e:
            raise SensorException(e, sys)
    
    def data_drift(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name:str):
        try:
            drift_report={}
            base_columns=base_df.columns
            current_columns=current_df.columns
            logging.info(f"checking of there is a drift in the base dataset and the current dataset")
            for base_column in base_columns:
                base_data,current_data=base_df[base_column],current_df[base_column]
                #logging.info(f" Hypothesis {base_column} :{base_data.dtype},{current_data.dtype}")
                same_distribution=ks_2samp(base_data,current_data)
                #print(f"type of value",type(same_distribution.pvalue))
                if same_distribution.pvalue>0.05 :
                    drift_report[base_column]={
                        'pvalue':float(same_distribution.pvalue),
                        'same_distribution':True
                    }
                else:
                    drift_report[base_column]={
                        'pvalue':float(same_distribution.pvalue),
                        'same_distribution':False
                    }
            self.validation_error[report_key_name]=drift_report
        except Exception as e:
            raise SensorException(e, sys)


    def initiate_data_validation(self) ->artifact_entity.DataValidationArtiFact:
        try:
            logging.info(f"======================Starting with data validation ==================")
            logging.info(f"Reading base dataframe")
            base_df=pd.read_csv(self.data_validation_config.base_file_path)
            logging.info(f"checking for columns -data validation {list(base_df.columns)}")
            logging.info("replace missing values in base dataframe")
            base_df.replace({"na":np.NAN},inplace=True)
            
            logging.info("drop null values column from base database")
            base_df=self.drop_missing_values_columns(df=base_df,report_key_name="missing_values_within_base_dataset")
            logging.info(f"reading train dataframe")
            
            train_df=pd.read_csv(self.data_ingestion_artifact.train_file_path)
            logging.info(f"checking for columns -data validation train df {list(train_df.columns)}")
           
            logging.info(f"train_file_path{self.data_ingestion_artifact.train_file_path}")
            logging.info(f"reading test dataframe")
            logging.info(f"test_file_path{self.data_ingestion_artifact.test_file_path}")
            
            test_df=pd.read_csv(self.data_ingestion_artifact.test_file_path)
            logging.info(f"checking for columns -data validation test df {list(test_df.columns)}")
           
            logging.info(f"drop missing values column in test and train database")
            train_df=self.drop_missing_values_columns(df=train_df,report_key_name="missing_values_within_train_dataset")
            test_df=self.drop_missing_values_columns(df=test_df,report_key_name="missing_values_within_test_dataset")
            exclude_columns=['class']
            base_df=utils.convert_column_float(df=base_df,exclude_columns=exclude_columns)
            train_df=utils.convert_column_float(df=train_df,exclude_columns=exclude_columns)
            test_df=utils.convert_column_float(df=test_df,exclude_columns=test_df)

            logging.info("checking if all the required columns exist in test and train database")
            train_column_status=self.is_required_columns_exist(base_df, train_df,report_key_name="missing_columns_within_train_dataset")
            test_column_status=self.is_required_columns_exist(base_df, test_df,report_key_name="missing_columns_within_test_dataset")

            logging.info("performing test to see if the distribution is intact in test and train database")
            if train_column_status:
                self.data_drift(base_df=base_df, current_df=train_df, report_key_name="data_drift_within_train_dataset")
            if test_column_status:
                 self.data_drift(base_df=base_df, current_df=test_df, report_key_name="data_drift_within_test_dataset")

            utils.write_yaml_file(file_path=self.data_validation_config.report_file_path,data=self.validation_error)
            data_validation_artifact=artifact_entity.DataValidationArtiFact(report_file_path=self.data_validation_config.report_file_path)
            return data_validation_artifact

        except Exception as e:
            raise SensorException(e, sys)
