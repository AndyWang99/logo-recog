from tensorflow.keras.preprocessing import image
from tensorflow import keras
import numpy as np
import os

def predict_class(model, images, show = True):
  food_list = ['samosa','pizza','omelette']
  for img in images:
    img = image.load_img(img, target_size=(300, 300))
    img = image.img_to_array(img)                    
    img = np.expand_dims(img, axis=0)         
    img /= 255.                                      

    pred = model.predict(img)
    index = np.argmax(pred)
    food_list.sort()
    pred_value = food_list[index]
    if show:
        print(pred_value)

def main():
	new_model = keras.models.load_model('7_epoch_model_trained_3class.hdf5')

	images = []
	images.append('pizza1.jpg')
	images.append('pizza2.jpg')
	images.append('pizza3.jpg')
	images.append('pizza4.jpg')
	images.append('pizza5.jpg')
	images.append('pizza6.jpg')
	images.append('pizza7.jpg')
	images.append('pizza8.jpg')
	images.append('pizza9.jpg')
	images.append('pizza10.jpg')
	
	images.append('samosa1.jpg')
	images.append('samosa2.jpg')
	images.append('samosa3.jpg')
	images.append('samosa4.jpg')
	images.append('samosa5.jpg')
	images.append('samosa6.jpg')
	images.append('samosa7.jpg')
	images.append('samosa8.jpg')
	images.append('samosa9.jpg')
	images.append('samosa10.jpg')
	
	images.append('omelette1.jpg')
	images.append('omelette2.jpg')
	images.append('omelette3.jpg')
	images.append('omelette4.jpg')
	images.append('omelette5.jpg')
	images.append('omelette6.jpg')
	images.append('omelette7.jpg')
	images.append('omelette8.jpg')
	images.append('omelette9.jpg')
	images.append('omelette10.jpg')


	predict_class(new_model, images, True)

main()