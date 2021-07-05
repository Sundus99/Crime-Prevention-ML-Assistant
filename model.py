import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

def teachable_machines_model(img,file):
	# Disable scientific notation for clarity
	np.set_printoptions(suppress=True)

	# Load the model
	model = tensorflow.keras.models.load_model(file,compile=False)#to use a saved model i fiugred u need to make compile=false

	# Create the array of the right shape to feed into the keras model
	# The 'length' or number of images you can put into the array is
	# determined by the first position in the shape tuple, in this case 1.
	data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

	# Replace this with the path to your image
	image = img

	#resize the image to a 224x224 with the same strategy as in TM2:
	#resizing the image to be at least 224x224 and then cropping from the center
	size = (224, 224)
	image = ImageOps.fit(image, size, Image.ANTIALIAS)

	#turn the image into a numpy array
	image_array = np.asarray(image)

	# display the resized image
	#image.show()

	# Normalize the image
	normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

	# Load the image into the array
	data[0] = normalized_image_array

	# run the inference
	prediction = model.predict(data)
	#with printing the prediction I got this numpy array: #[[0.49391878 0.5060812]] which are decimal values
	#for class 0 and 1 predictions (These values were shown in % in teachable machines for unarmed.jpg)
	#But what I am interested in is knowing which class it is leaning towards the most (Class 0: Armed or Class 1: Unarmed)
	#this can be accomplished with the following numpy function:
	return np.argmax(prediction)
