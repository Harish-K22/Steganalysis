import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
import efficientnet.tfkeras as efn
from pathlib import Path
from Steganalysis.entity.config_entity import PrepareBaseModelConfig

AUTO = tf.data.experimental.AUTOTUNE

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.strategy = self._get_strategy()
        self.model = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def _get_strategy(self):
        """Detect TPU or GPU strategy for distributed training."""
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            print(f'Running on TPU {tpu.master()}')
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            return tf.distribute.experimental.TPUStrategy(tpu)
        except ValueError:
            print("TPU not detected, using default strategy.")
            return tf.distribute.get_strategy()

    def get_base_model(self):
        with self.strategy.scope():
            if self.config.model_type == "EfficientNetB3":
                self.model = tf.keras.Sequential([
                    efn.EfficientNetB3(
                        input_shape=tuple(self.config.params_image_size),
                        weights=self.config.params_weights,
                        include_top=self.config.params_include_top
                    ),
                    layers.GlobalAveragePooling2D(),
                    layers.Dense(1, activation="sigmoid")
                ])
            else:
                raise ValueError(f"Model type {self.config.model_type} is not supported.")
            
            self.compile_model()
            self.model.summary()
            
            # Save model here
            self.save_model(self.config.base_model_path)  # Save the base model file

    def compile_model(self):
        """Compiles the model with configurations from params.yaml."""
        self.model.compile(
            optimizer=tf.keras.optimizers.get(self.config.optimizer),
            loss=self.config.loss_function,
            metrics=self.config.metrics
        )

    def prepare_data(self):
        """Prepares balanced data paths and splits them into train, validation, and test sets."""
        
        def get_paths(directory, start_index, end_index):
            """Helper function to get image paths from a directory within a range."""
            return [os.path.join(directory, filename) for filename in sorted(os.listdir(directory))[start_index:end_index]]

        # Define paths for each class
        cover_paths = np.array(get_paths('/kaggle/input/alaska2-image-steganalysis/Cover', 0, 75000))
        jmipod_paths = get_paths('/kaggle/input/alaska2-image-steganalysis/JMiPOD', 0, 25000)
        juniward_paths = get_paths('/kaggle/input/alaska2-image-steganalysis/JUNIWARD', 25000, 50000)
        uerd_paths = get_paths('/kaggle/input/alaska2-image-steganalysis/UERD', 50000, 75000)

        # Labels: Cover = 0, Stego = 1
        cover_labels = np.array([0] * len(cover_paths))
        stego_paths = np.array(jmipod_paths + juniward_paths + uerd_paths)
        stego_labels = np.array([1] * len(stego_paths))

        # Combine and shuffle
        img_paths = np.concatenate((cover_paths, stego_paths), axis=None)
        img_labels = np.concatenate((cover_labels, stego_labels), axis=None)

        # Split into train, test, validation sets
        X_train_paths, X_test_paths, y_train, y_test = train_test_split(img_paths, img_labels, test_size=0.15, shuffle=self.config.shuffle_data)
        X_train_paths, X_val_paths, y_train, y_val = train_test_split(X_train_paths, y_train, test_size=0.15, shuffle=self.config.shuffle_data)

        self.train_ds = self.build_dataset(X_train_paths, y_train)
        self.test_ds = self.build_dataset(X_test_paths, y_test)
        self.val_ds = self.build_dataset(X_val_paths, y_val)

    def data_augment(self, image, label):
        """Applies data augmentation."""
        image = tf.image.random_flip_left_right(image)
        return image, label  

    def decode_img(self, path, label):
        """Reads and decodes image from path, resizes, and normalizes."""
        bits = tf.io.read_file(path)
        image = tf.image.decode_jpeg(bits, channels=3)
        image = tf.image.resize(image, self.config.params_image_size[:2]) / 255.0
        return image, label

    def build_dataset(self, X, y):
        """Builds a tf.data.Dataset for training, validation, or testing."""
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.map(self.decode_img, num_parallel_calls=AUTO)
        dataset = dataset.map(self.data_augment, num_parallel_calls=AUTO)
        dataset = dataset.batch(self.config.batch_size).prefetch(AUTO)
        return dataset

    def save_model(self, path=None):
        """Saves the model to the specified path."""
        path = path or self.config.base_model_path
        self.model.save(path)
        print(f"Model saved at {path}")
