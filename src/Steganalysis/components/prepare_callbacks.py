import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from Steganalysis.entity.config_entity import PrepareCallbacksConfig

class PrepareCallbacks:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config
        self.model_checkpoint = None
        self.tensorboard = None

    def create_model_checkpoint_callback(self):
        """Creates ModelCheckpoint callback"""
        self.model_checkpoint = ModelCheckpoint(
            filepath=self.config.checkpoint_model_filepath,
            save_best_only=True
        )

    def create_tensorboard_callback(self):
        """Creates TensorBoard callback"""
        self.tensorboard = TensorBoard(
            log_dir=self.config.tensorboard_root_log_dir
        )

    def get_callbacks(self):
        """Creates all callbacks and returns them as a list"""
        self.create_model_checkpoint_callback()
        self.create_tensorboard_callback()
        return [self.model_checkpoint, self.tensorboard]
