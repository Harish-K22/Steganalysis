from Steganalysis.config.configuration import ConfigurationManager
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from Steganalysis import logger
import os

STAGE_NAME = "Model Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        model_training_config = config_manager.get_model_training_config()
        
        model = load_model(model_training_config.model_path)
        data_gen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)  # Adjust as per data needs
        
        data_dir = "artifacts/data_ingestion" 
        
        train_generator = data_gen.flow_from_directory(
            directory=model_training_config.data_path,  # This should map to `artifacts/data_ingestion`
            target_size=model_training_config.image_size[:2],
            batch_size=model_training_config.batch_size,
            class_mode='binary',
            shuffle=model_training_config.shuffle_data
        )
        
        val_generator = data_gen.flow_from_directory(
            directory=data_dir,
            target_size=model_training_config.image_size[:2],
            batch_size=model_training_config.batch_size,
            class_mode="binary",
            subset="validation"
        )
        
        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=os.path.join(model_training_config.callbacks_dir, "tensorboard_log")),
            tf.keras.callbacks.ModelCheckpoint(filepath=model_training_config.model_path, save_best_only=True)
        ]
        
        model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=model_training_config.epochs,
            callbacks=callbacks
        )
        
        logger.info(f">>>>>> Model training completed. Model saved to {model_training_config.model_path} <<<<<<")
