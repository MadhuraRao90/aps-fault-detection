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
    

class ModelTrainerArtiFact:...
class ModelEvaluationArtiFact:...
class ModelPusherArtiFact:...

