import tensorflow as tf

converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file(
	'6_epoch_model_trained_3class.hdf5',
    input_shapes = {"input_1": [32, 300, 300, 3]},
	)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
