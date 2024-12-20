from Steganalysis.constants import *
from Steganalysis.utils.common import read_yaml, create_directories
from Steganalysis.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig, PrepareCallbacksConfig, ModelTrainingConfig
from pathlib import Path
from box import ConfigBox

# Define default paths for config and params files
CONFIG_FILE_PATH = 'E:/Projects/Steganalysis/config/config.yaml'
PARAMS_FILE_PATH = 'E:/Projects/Steganalysis/params.yaml'

class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        # Load configuration and parameters with ConfigBox for easy access
        self.config = ConfigBox(read_yaml(Path(config_filepath)))
        self.params = ConfigBox(read_yaml(Path(params_filepath)))
        
        # Create root directory for artifacts
        create_directories([Path(self.config.artifacts_root)])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        # Ensure data ingestion directories are created
        create_directories([Path(config.root_dir)])

        # Create and return DataIngestionConfig object
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )

        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        model_config = self.params.model
        training_config = self.params.training

        # Ensure directories are created for model preparation
        create_directories([self.config.prepare_base_model.root_dir])

        # Prepare and return PrepareBaseModelConfig object with model configurations
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(self.config.prepare_base_model.root_dir),
            base_model_path=Path(self.config.prepare_base_model.base_model_path),
            updated_base_model_path=Path(self.config.prepare_base_model.updated_base_model_path),
            params_image_size=tuple(model_config.input_shape),
            params_include_top=model_config.include_top,
            params_weights=model_config.weights,
            params_classes=model_config.classes,
            model_type=model_config.model_type,
            optimizer=model_config.optimizer,
            loss_function=model_config.loss_function,
            metrics=model_config.metrics,
            batch_size=training_config.batch_size,
            shuffle_data=training_config.shuffle_data
        )

        return prepare_base_model_config

    def get_prepare_callbacks_config(self) -> PrepareCallbacksConfig:
        create_directories([self.config.prepare_callbacks.root_dir])
        
        prepare_callbacks_config = PrepareCallbacksConfig(
            root_dir=Path(self.config.prepare_callbacks.root_dir),
            tensorboard_root_log_dir=Path(self.config.prepare_callbacks.tensorboard_root_log_dir),
            checkpoint_model_filepath=Path(self.config.prepare_callbacks.checkpoint_model_filepath)
        )
        return prepare_callbacks_config
    
    def get_model_training_config(self) -> ModelTrainingConfig:
        training_config = self.params.training
        model_training_config = ModelTrainingConfig(
            model_path=Path(self.config.prepare_base_model.updated_base_model_path),
            epochs=training_config.epochs,
            batch_size=training_config.batch_size,
            shuffle_data=training_config.shuffle_data,
            image_size=tuple(self.params.model.input_shape),
            root_dir=Path(self.config.training.root_dir),
            callbacks_dir=Path(self.config.prepare_callbacks.root_dir),
            data_path=Path(self.config.data_ingestion.root_dir)  # Add data_path from config
        )
        return model_training_config
