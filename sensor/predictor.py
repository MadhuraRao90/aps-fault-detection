from sensor.exception import SensorException
import os,sys
from sensor.entity.config_entity import TRANSFORMER_OBJECT_FILE_NAME,MODEL_FILE_NAME,TARGET_ENCODER_OBJECT_FILE_NAME
from glob import glob
class ModelResolver:

    def __init__(self,model_registry:str="saved_models",transformer_model_dir="transformer",
    target_encoder_dir_name="target_encoder",model_dir_name="model"):
        self.model_registry=model_registry
        os.makedirs(self.model_registry,exist_ok=True)
        self.transformer_model_dir=transformer_model_dir
        self.target_encoder_dir_name=target_encoder_dir_name
        self.model_dir_name=model_dir_name

    def get_latest_dir_path(self):
        try:
            dir_names=os.listdir(self.model_registry)
            if len(dir_names)==0:
                return None
            dir_names=list(map(int,dir_names))
            latest_folder_name=max(dir_names)
            return os.path.join(self.model_registry,f"{latest_folder_name}")
        except Exception as e:
            raise SensorException(e,sys)

    def get_latest_model_path(self):
        try:
            latest_dir=self.get_latest_dir_path()
            if latest_dir==None:
                #raise Exception(f"Model is not available")
                return None
            return os.path.join(latest_dir,self.model_dir_name,MODEL_FILE_NAME)
        except Exception as e:
            raise SensorException(e, sys)


    def get_latest_transformer_path(self):
        try:
            latest_dir=self.get_latest_dir_path()
            if latest_dir==None:
                #raise Exception(f"Transformer is not available")
                return None
            return os.path.join(latest_dir,self.transformer_model_dir,TRANSFORMER_OBJECT_FILE_NAME)

        except Exception as e:
            raise SensorException(e, sys)

    def get_latest_target_encoder_path(self):
        try:
            latest_dir=self.get_latest_dir_path()
            if latest_dir==None:
                #raise Exception(f"Encoder is not available")
                return None
            return os.path.join(latest_dir,self.target_encoder_dir_name,TARGET_ENCODER_OBJECT_FILE_NAME)
        except Exception as e:
            raise SensorException(e, sys)

    def get_latest_save_dir_path(self):
        try:
            dir_names=os.listdir(self.model_registry)
            if len(dir_names)==0:
                return os.path.join(self.model_registry,f"{0}")
            dir_names=list(map(int,dir_names))
            latest_folder_name=max(dir_names)
            new_latest=latest_folder_name+1
            return os.path.join(self.model_registry,f"{new_latest}")
        except Exception as e:
            raise SensorException(e,sys)

    def get_latest_save_model_path(self):
        try:
            latest_dir=self.get_latest_save_dir_path()
            
            return os.path.join(latest_dir,self.model_dir_name,MODEL_FILE_NAME)
        except Exception as e:
            raise SensorException(e, sys)
    def get_latest_save_target_encoder_path(self):
        try:
            latest_dir=self.get_latest_save_dir_path()
            
            return os.path.join(latest_dir,self.target_encoder_dir_name,TARGET_ENCODER_OBJECT_FILE_NAME)
        except Exception as e:
            raise SensorException(e, sys)
    def get_latest_save_transformer_path(self):
        try:
            latest_dir=self.get_latest_save_dir_path()
            
            return os.path.join(latest_dir,self.transformer_model_dir,TRANSFORMER_OBJECT_FILE_NAME)
        except Exception as e:
            raise SensorException(e, sys)

    
