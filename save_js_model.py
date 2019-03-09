import tensorflowjs as tfjs
from tensorflow import keras

model = keras.models.load_model('6_epoch_model_trained_3class.hdf5')

tfjs.converters.save_keras_model(model, '.')
