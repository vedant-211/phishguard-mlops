import os
import sys
import json
import pymongo
import pandas as pd
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

class NetworkDataExtract:
    def __init__(self):
        pass

    def csv_to_json_converter(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def insert_data_mongodb(self, records, database, collection):
        try:
            mongo_db_url = os.getenv("MONGO_DB_URL")
            if not mongo_db_url:
                print("MONGO_DB_URL is not set. Skipping MongoDB push.")
                return 0
            mongo_client = pymongo.MongoClient(mongo_db_url)
            db = mongo_client[database]
            col = db[collection]
            col.insert_many(records)
            return len(records)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

if __name__ == "__main__":
    FILE_PATH = "data/phisingData.csv"
    DATABASE = "AI"
    COLLECTION = "NetworkData"
    network_obj = NetworkDataExtract()
    records = network_obj.csv_to_json_converter(file_path=FILE_PATH)
    num_records = network_obj.insert_data_mongodb(records, DATABASE, COLLECTION)
    print(f"Successfully inserted {num_records} records into MongoDB Atlas.")
