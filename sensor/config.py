import pymongo
import json
import pandas as pd
from dataclasses import dataclass
import os
#client = pymongo.MongoClient("mongodb://localhost:27017/neurolabDB")
client=pymongo.MongoClient("mongodb+srv://madhura:Maddy2809@cluster0.jx1ws.mongodb.net/?retryWrites=true&w=majority")

@dataclass
class EnvironmentalVariable:
    mongo_db_url:str=os.getenv("MONGO_DB_URL")
    aws_access_key:str=os.getenv("AWS_ACCESS_KEY")
    aws_access_secret_key:str=os.getenv("AWS_SECRET_ACCESS_KEY")

env_var=EnvironmentalVariable()

mongo_client=pymongo.MongoClient(env_var.mongo_db_url)
