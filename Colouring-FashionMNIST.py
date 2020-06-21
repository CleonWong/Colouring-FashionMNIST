import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage.transform import resize


image = load_img(path="Images/Train/0A9kTN.jpg")
image = img_to_array(image)
image = resize(image=image, output_shape=(28, 28, 3))
image = np.array(image, dtype = float)

print(image.shape)




plt.imshow(image)
plt.show()
