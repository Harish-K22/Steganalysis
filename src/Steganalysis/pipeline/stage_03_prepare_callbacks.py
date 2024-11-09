from Steganalysis.config.configuration import ConfigurationManager
from Steganalysis.components.prepare_callbacks import PrepareCallbacks
from Steganalysis import logger

STAGE_NAME = "Prepare Callbacks"

class PrepareCallbacksTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        prepare_callbacks_config = config_manager.get_prepare_callbacks_config()
        prepare_callbacks = PrepareCallbacks(config=prepare_callbacks_config)
        callbacks = prepare_callbacks.get_callbacks()
        return callbacks

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareCallbacksTrainingPipeline()
        callbacks = obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
