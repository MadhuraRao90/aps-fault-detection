from sensor.exception import SensorException
from sensor.logger import logging
import sys,os
from sensor.utils import get_collection_as_dataframe
from sensor.entity import config_entity,artifact_entity
from sensor.components import data_ingestion
from sensor.components import data_validation
from sensor.components import data_transformation
from sensor.components import model_trainer
from sensor.components import model_evaluation
from sensor.components import model_pusher
import sensor.utils 


def start_training_pipeline():
     try:
          logging.info(f"=============this is where we have done data ingestion==============")
          training_pipeline_config=config_entity.TrainingPipelineConfig()
          #data ingestion
          data_ingestion_config=config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
          #print(data_ingestion_config.to_dict())
          data_ing=data_ingestion.DataIngestion(data_ingestion_config=data_ingestion_config)
          #data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
          data_ingestion_artifact=data_ing.initiate_data_ingestion()
          #print(f"test file path{data_ingestion_artifact.test_file_path}{data_ingestion_artifact.train_file_path}")
     except Exception as e:
          raise SensorException(e,sys)
     try:          
          logging.info(f"=============this is where we have done data validation==============")
          # this is where we have done data validation
          data_validation_config=config_entity.DataValidationConfig(training_pipeline_config=training_pipeline_config)
          data_val=data_validation.DataValidation(data_validation_config=data_validation_config,data_ingestion_artifact=data_ingestion_artifact) 
          data_validation_artifact=data_val.initiate_data_validation()
     except Exception as e:
          raise SensorException(e,sys)
     try:
          logging.info(f"=============this is where we have done data transformation==============")
          # this is where we have done data transformation
          data_transformation_config=config_entity.DataTransformationConfig(training_pipeline_config=training_pipeline_config)
          data_transf=data_transformation.DataTransformation(data_transformation_config=data_transformation_config, data_ingestion_artifact=data_ingestion_artifact)
          data_transformation_artifact=data_transf.initiate_data_transformation()
     except Exception as e:
          raise SensorException(e,sys)   
     try:
          logging.info(f"=============this is where we have done model training==============")
         
     # this is where we done model training
          model_trainer_config=config_entity.ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
          model_train=model_trainer.ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
          model_trainer_artifact=model_train.initiate_model_trainer()
     except Exception as e:
          raise SensorException(e,sys) 
     try:
          logging.info(f"=============this is where we have done model evaluation==============")
     
          # This where model evaluation is done to see which model - previous or the current model performs better
          model_evaluation_config=config_entity.ModelEvaluationConfig(training_pipeline_config=training_pipeline_config)
          model_eval=model_evaluation.ModelEvaluation(model_eval_config=model_evaluation_config, data_ingestion_artifact=data_ingestion_artifact, data_transformation_artifact=data_transformation_artifact, model_trainer_artifact=model_trainer_artifact)
          model_evaluation_artifact=model_eval.initiate_model_eval()
     except Exception as e:
          raise SensorException(e,sys)
     try: 
          logging.info(f"=============this is where we have model pusher==============")
     
          # model pusher
          model_pusher_config=config_entity.ModelPusherConfig(training_pipeline_config=training_pipeline_config)
          model_push=model_pusher.ModelPusher(model_pusher_config=model_pusher_config,
          data_transformation_artifact=data_transformation_artifact, model_trainer_artifact=model_trainer_artifact)
          model_pusher_artifact=model_push.initiate_model_pusher()
          

     except Exception as e:
          raise SensorException(e,sys)