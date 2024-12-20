import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from tensorflow.keras.layers import Dropout  # Import Dropout
from tensorflow.keras.utils import get_custom_objects

# Register custom layers/activations if needed
class FixedDropout(Dropout):
    def __init__(self, rate, seed=None, **kwargs):
        super(FixedDropout, self).__init__(rate, seed=seed, **kwargs)

get_custom_objects().update({
    "FixedDropout": FixedDropout,
    "swish": tf.keras.activations.swish
})

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load model - E:\Projects\Steganalysis\artifacts\Private\model_checkpoint.keras
        model_path = os.path.abspath(os.path.join("artifacts", "Private", "model_checkpoint.keras"))
        model = load_model(model_path)

        # Load and preprocess image
        test_image = image.load_img(self.filename, target_size=(512, 512))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0) / 255.0

        # Get prediction
        result = model.predict(test_image)
        if result[0] >= 0.5:
            prediction = 'Stego'
        else:
            prediction = 'Cover'

        return [{"image": prediction}]
