#Importing required libraries

from tensorflow.keras.layers import GlobalMaxPool2D,AveragePooling2D,Dropout,Flatten,Dense,Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from imutils import paths
import numpy as np
import argparse
import os

# Use argeparse for command line instructions
ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True)
args=vars(ap.parse_args())
lr=0.0004
epochs=10
Batch_Size=128
image_size=(224,224)

#----------------Loading images from our dataset and doing required image processing and making it compatible with the input MobileNet expects-----------

#Accessing Dataset
imagePaths = list(paths.list_images(args["dataset"]))
dataset=[]
labels=[]

# loop over the image paths to create the dataset
for imagePath in imagePaths:
	
	label=imagePath.split(os.path.sep)[-2]
	image=load_img(imagePath, target_size=image_size)
	image=img_to_array(image)
	image=preprocess_input(image)
	dataset.append(image)
	labels.append(label)


dataset= np.array(dataset)
labels = np.array(labels)

le=LabelEncoder()
labels=le.fit_transform(labels)
labels=to_categorical(labels)


#Splitting the data for training and validation
(X_train,X_valid,y_train,y_valid)=train_test_split(dataset,labels,
	test_size=0.20,stratify=labels,random_state=42)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

#------------------Building the model--------------
#Using MobileNet as our base model for fine tuning . We will leave out the top layer
# and ourselves add a few layers to make it compatible with the output we expect
baseModel=MobileNetV2(weights="imagenet",include_top=False,
	input_tensor=Input(shape=(image_size[0],image_size[1],3)))

output=baseModel.output
output=AveragePooling2D(pool_size=(7,7))(output)
output=Flatten()(output)
output=Dense(128,activation="relu")(output)
output=Dropout(0.4)(output)
output=Dense(2,activation="softmax")(output)

model=Model(inputs=baseModel.input,outputs=output)
for layer in baseModel.layers:
	layer.trainable = False

optim=Adam(lr=lr)
model.compile(loss="binary_crossentropy", optimizer=optim,metrics=["accuracy"])
#------Training the model on our dataset---------------
History=model.fit(
	aug.flow(X_train,y_train,batch_size=Batch_Size),
	steps_per_epoch=len(X_train)//Batch_Size,
	validation_data=(X_valid,y_valid),
	validation_steps=len(X_valid)//Batch_Size,
	epochs=epochs)
# Saving the final model
model.save("mask_detector_model.model",save_format="h5")

