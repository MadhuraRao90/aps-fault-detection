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
from sensor.pipeline import training_pipeline
from sensor.pipeline.batch_pieline import batch_prediction

if __name__=="__main__":  
     try:

          #training_pipeline.start_training_pipeline()
          output_file=batch_prediction("/config/workspace/aps_failure_training_set1.csv")
          print(output_file)
     except Exception as e:
          raise SensorException(e,sys)     
          

          