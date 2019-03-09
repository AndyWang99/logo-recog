import os
import glob
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

images = glob.glob("data/test/logo_starbucks/*.jpg")
for image in images:
    with open(image, 'rb') as file:
        img = Image.open(file)
        # now do something with the image
        
        x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

        # the .flow() command below generates batches of randomly transformed images
        # and saves the results to the `preview/` directory
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir='modified_images', save_prefix='mod_img', save_format='jpg'):
            i += 1
            if i > 19:
                break  # otherwise the generator would loop indefinitely