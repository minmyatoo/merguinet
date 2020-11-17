# Windows
# python minmyat_flowers_dataset.py --dataset flowers_dataset --output test.png
# python minmyat_flowers_dataset.py --dataset flowers_dataset --output test3.png
# python minmyat_flowers_dataset.py --dataset flowers_dataset --output test82.png

# Sklearn Libraries
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from imutils import paths

# Local modules
# https://github.com/CzJakethedog/preprocessing (BUlk lists of Preprocessing files)
from mmo.imagetoarraypreprocessor import ImageToArrayPreprocessor
from mmo.Aspectawrepre import AspectAwarePreprocessor
from mmo.simpledatasetloader import SimpleDatasetLoader
from mmo.model.minivggnet import MerguiNet

import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

# construct the argument parse and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='Data Path to input dataset')
ap.add_argument("-o", "--output", required=True, help="path to the output loss/accuracy plot")
# ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())

# grab the list of images and extract class label names from image paths
print('[INFO] loading images ...')
imagePaths = list(paths.list_images(args['dataset']))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

# initialize image preprocessors
aap = AspectAwarePreprocessor(64, 64)
iap = ImageToArrayPreprocessor()

# load dataset from disk and scale raw pixel intensities to range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype('float') / 255.0

# splitting data into training ans testing set
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# convert labels from integers to Vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# constructing image generator for Data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode='nearest')

# initializing the optimizer and model
print('[INFO] compiling model ...')
opt = SGD(lr=0.025)  # Get 80
# add Momentum
# opt = SGD(lr=0.025,momentum=0.9) #Get 76

# add Option #Get 81
# opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)

# add Option #Get 81
# opt = SGD(lr=0.025, decay=0.01 / 40, momentum=0.9, nesterov=True)

# Creating Model
model = MerguiNet.build(width=64, height=64, depth=3, classes=len(classNames))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# train network
print('[INFO] training network ...')
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=32), validation_data=(testX, testY),
                        steps_per_epoch=len(trainX) // 32, epochs=100, verbose=1)

# evaluate the model
print('[INFO] evaluating the network ...')
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))

# Check the test loss and test accuracy
score = model.evaluate(testX, testY, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1] * 100)

print("[INFO] serializing network...")
# model.save(args["model"])

# model.save('merguimodel.hdf5')
# Print Keys
# print(H.history)


# plot training loss and accuracy
# plt.style.use("ggplot")
plt.style.use("classic")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])
