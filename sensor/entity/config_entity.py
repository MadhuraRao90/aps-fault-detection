
import os,sys
from datetime import datetime
from sensor.exception import SensorException
FILE_NAME="sensor.csv"
TRAIN_FILE_NAME="train.csv"
TRANSFORMER_OBJECT_FILE_NAME='tranformer.pkl'
TARGET_ENCODER_OBJECT_FILE_NAME='target_encoder.pkl'
TEST_FILE_NAME="test.csv"
MODEL_FILE_NAME="model.pkl"


class TrainingPipelineConfig:
    try:
        def __init__(self):
            self.artifact_dir=os.path.join(os.getcwd(),"artifact",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
    except Exception as e:
        raise SensorException(e,sys)
class DataIngestionConfig:
    try:
        def __init__(self,training_pipeline_config:TrainingPipelineConfig):
            self.database_name="aps"
            self.collection_name="sensor"
            self.data_ingestion_dir=os.path.join(training_pipeline_config.artifact_dir,"data_ingestion")
            self.feature_store_file_path=os.path.join(self.data_ingestion_dir,"feature_store",FILE_NAME)
            self.train_file_path=os.path.join(self.data_ingestion_dir,"dataset",TRAIN_FILE_NAME)
            self.test_file_path=os.path.join(self.data_ingestion_dir,"dataset",TEST_FILE_NAME)
            self.test_size=0.2
    except Exception as e:
        logging.debug(f"{SensorException(e, sys)}")
        raise SensorException(e, sys)    
    def to_dict(self,)-> dict:
        try:
            return self.__dict__
        except Exception as e:
            logging.debug(f"{SensorException(e, sys)}")
            raise SensorException(e,sys)
            

        
class DataValidationConfig:
    try:

        def __init__(self,training_pipeline_config:TrainingPipelineConfig):
            self.data_validation_dir=os.path.join(training_pipeline_config.artifact_dir,"data_validation")
            self.report_file_path=os.path.join(self.data_validation_dir,"report.yaml")
            self.missing_threshold=0.2
            self.base_file_path=os.path.join("aps_failure_training_set1.csv")
            
            #print(f"{self.data_validation_dir}")
    except Exception as e:



        raise SensorException(e, sys) 
           
class DataTransformationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir=os.path.join(training_pipeline_config,"data_transformation")
        self.tranform_object_path=os.path.join(self.data_transformation_dir,"transformer",TRANSFORMER_OBJECT_FILE_NAME)
        self.transformed_train_path=os.path.join(self.data_transformation_dir,"transformed",TRAIN_FILE_NAME.replace("csv", "npz"))
        self.transformed_test_path=os.path.join(self.data_transformation_dir,"transformed",TEST_FILE_NAME.replace("csv", "npz"))
        self.target_column_enoding=os.path.join(self.data_transformation_dir,"target_encoder",TARGET_ENCODER_OBJECT_FILE_NAME)

class ModelTrainerConfig:...
class ModelEvaluationConfig:...
class ModelPusherConfig:...




TARGET_COLUMN='class'
