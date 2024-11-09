import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from Steganalysis import logger

from Steganalysis.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from Steganalysis.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline

STAGE_NAME = 'Data Ingestion stage'
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Prepare base model" 

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e