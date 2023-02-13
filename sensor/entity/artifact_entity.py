from dataclasses import dataclass
@dataclass()
class DataIngestionArtiFact:
    feature_store_file_path:str
    train_file_path:str
    test_file_path:str 
    
@dataclass()  
class DataValidationArtiFact:
    report_file_path:str
@dataclass()
class DataTransformationArtiFact:
    tranform_object_path:str
    transformed_train_path:str
    transformed_test_path:str
    target_encoder_path:str
    
    
@dataclass()
class ModelTrainerArtiFact:
    model_trainer:str
    f1_train_score:float
    f1_test_score:float

@dataclass()
class ModelEvaluationArtiFact:
    is_model_accepted:bool
    improved_accuracy:float 
@dataclass()
class ModelPusherArtiFact:
    pusher_model_dir:str
    saved_model_sir=str
