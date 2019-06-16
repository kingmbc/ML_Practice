#!pip install -q tensorflow_data_validation

import tensorflow as tf
import tensorflow_data_validation as tfdv

tf.logging.set_verbosity(tf.logging.ERROR)
print('TFDV version: {}'.format(tfdv.version.__version__))