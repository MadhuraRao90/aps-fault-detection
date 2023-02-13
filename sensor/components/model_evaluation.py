from sensor.predictor import ModelResolver
from sensor.entity import config_entity,artifact_entity
from sensor.exception import SensorException
import sys,os
from sensor.logger import logging
from sensor.utils import load_object
from sklearn.metrics import f1_score
import pandas as pd
from sensor.entity.config_entity import TARGET_COLUMN
class  ModelEvaluation:
    def __init__(self,
    model_eval_config:config_entity.ModelEvaluationConfig,data_ingestion_artifact:artifact_entity.DataIngestionArtiFact,
    data_transformation_artifact:artifact_entity.DataTransformationArtiFact,
    model_trainer_artifact:artifact_entity.ModelTrainerArtiFact):
        try:
            self.model_eval_config=model_eval_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact
            self.model_resolver=ModelResolver()
        except Exception as e:
            raise SensorException(e,sys)
    def initiate_model_eval(self,)->artifact_entity.ModelEvaluationArtiFact:
        try:
            # if save model folder has model then we will compare which model is best 
            # trained or the model from saved model

            latest_dir_path=self.model_resolver.get_latest_model_path()
            if latest_dir_path==None:
                model_eval_artifact=artifact_entity.ModelEvaluationArtiFact(is_model_accepted=True, improved_accuracy=None)
                logging.info(f"model evaluation artifact {model_eval_artifact}")
                return model_eval_artifact
            # getting the latest transformer path, model path and encoder path
            logging.info(f"getting the latest transformer path, model path and encoder path")
            transformer_path=self.model_resolver.get_latest_transformer_path()
            model_path=self.model_resolver.get_latest_model_path()
            encoder_path=self.model_resolver.get_latest_target_encoder_path()

            #loading objects 
            logging.info(f"loading the previously trained objects")
            transformer=load_object(file_path=transformer_path)
            ip_feature=list(transformer.feature_names_in_)
            logging.info(f"input features_model_eval {ip_feature}")
            model=load_object(file_path=model_path)
            target_encoder=load_object(file_path=encoder_path)

            # currently trained model objects
            logging.info(f"loading the current trained objects")
           
            current_tranformer=load_object(file_path=self.data_transformation_artifact.tranform_object_path)
            current_model=load_object(file_path=self.model_trainer_artifact.model_trainer)
            current_target_encoder=load_object(file_path=self.data_transformation_artifact.target_encoder_path)
            current_ip_features=list(current_tranformer.feature_names_in_)
            test_df=pd.read_csv(self.data_ingestion_artifact.test_file_path)
            target_df=test_df[TARGET_COLUMN]
            y_true=target_encoder.transform(target_df)
            #accuracy using previously trained model
            logging.info(f"features used during fit{transformer.feature_names_in_}")
            logging.info(f"features in current transformer {current_tranformer.feature_names_in_}")
            logging.info(f"features that is actually present in test {test_df.drop(TARGET_COLUMN,axis=1).columns}")
            input_arr=transformer.transform(test_df.drop(TARGET_COLUMN,axis=1))
            y_pred=model.predict(input_arr)
            
            previous_model_score=f1_score(y_true=y_true,y_pred=y_pred)
            logging.info(f"accuracy using previously trained model: {previous_model_score}")
            #print(f"prediction using previous model: {target_encoder.inverse_transform(y_pred[:5])}")

            # accuracy using current trained model
            y_true_current=current_target_encoder.transform(target_df)
            input_arr_current=current_tranformer.transform(test_df.drop(TARGET_COLUMN,axis=1))
            y_pred_current=current_model.predict(input_arr_current)

            current_model_score=f1_score(y_true=y_true_current,y_pred=y_pred_current)  
            logging.info(f"accuracy using current trained model: {current_model_score}")
                  
            #print(f"prediction using current model: {current_target_encoder.inverse_transform(y_pred_current[:5])}")

            if current_model_score <= previous_model_score:
                logging.info("Current trained model score is not better than previous model ")
                raise Exception("Current trained model score is not better than previous model ")
            
            model_eval_artifact=artifact_entity.ModelEvaluationArtiFact(is_model_accepted=True,improved_accuracy=current_model_score-previous_model_score)
            logging.info(f"is model accepted : {model_eval_artifact.is_model_accepted}")
            logging.info(f"Model evaluation artifact :{model_eval_artifact}")
            return model_eval_artifact
        except Exception as e:
            raise SensorException(e, sys)