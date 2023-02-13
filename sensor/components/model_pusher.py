

from sensor.predictor import ModelResolver
from sensor.entity.config_entity import ModelPusherConfig
from sensor.entity.artifact_entity import DataTransformationArtiFact,ModelTrainerArtiFact,ModelPusherArtiFact
from sensor.exception import SensorException
import os,sys
from sensor.logger import logging
from sensor import utils
class ModelPusher:
    def __init__(self,model_pusher_config:ModelPusherConfig,
    data_transformation_artifact:DataTransformationArtiFact,model_trainer_artifact:ModelTrainerArtiFact):
        try:
            self.model_pusher_config=model_pusher_config
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact
            self.model_resolver=ModelResolver(model_registry=self.model_pusher_config.saved_model_dir)
        except Exception as e:
            raise SensorException(e,sys)

    def initiate_model_pusher(self,)->ModelPusherArtiFact:
        try:
            # load object 
            logging.info("loading transformer model and target encoder")
            transformer=utils.load_object(file_path=self.data_transformation_artifact.tranform_object_path)
            model=utils.load_object(file_path=self.model_trainer_artifact.model_trainer) 
            encoder=utils.load_object(file_path=self.data_transformation_artifact.target_encoder_path )

            # model pusher dir
            logging.info("save the models in the saved models directory inside artifacts")
            utils.save_object(file_path=self.model_pusher_config.pusher_model_path, obj=model)
            utils.save_object(file_path=self.model_pusher_config.pusher_transformer_path,obj=transformer)
            utils.save_object(file_path=self.model_pusher_config.pusher_target_encoder_path, obj=encoder)

            # saved model dir
            logging.info("saving the models in saved models folder")
            saved_model_path=self.model_resolver.get_latest_save_model_path()
            saved_transformer_path=self.model_resolver.get_latest_save_transformer_path()
            saved_encoder_path=self.model_resolver.get_latest_save_target_encoder_path()
            utils.save_object(file_path=saved_model_path, obj=model)
            utils.save_object(file_path=saved_transformer_path, obj=transformer)
            utils.save_object(file_path=saved_encoder_path, obj=encoder)


        
        except Exception as e:
            raise SensorException(e, sys)
