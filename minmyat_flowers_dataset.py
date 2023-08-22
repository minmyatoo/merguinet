# 🌼 Windows Commands:
# 🏁 Run the script with different output files
# 🐍 python minmyat_flowers_dataset.py --dataset flowers_dataset --output test.png
# 🐍 python minmyat_flowers_dataset.py --dataset flowers_dataset --output test3.png
# 🐍 python minmyat_flowers_dataset.py --dataset flowers_dataset --output test82.png

# Import necessary packages 📦
import numpy as np
import cv2
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model
from imutils import paths

# Import local modules 📂
from mmo.imagetoarraypreprocessor import ImageToArrayPreprocessor
from mmo.aspectawrepre import AspectAwarePreprocessor
from mmo.simpledatasetloader import SimpleDatasetLoader
from mmo.model.minivggnet import MerguiNet

def main():
    # 📟 Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', required=True, help='Path to the input dataset')
    ap.add_argument("-o", "--output", required=True, help="Path to the output loss/accuracy plot")
    args = vars(ap.parse_args())

    # 🖼️ Load images and extract class label names from image paths
    print('[INFO] Loading images ...')
    imagePaths = list(paths.list_images(args['dataset']))
    classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
    classNames = [str(x) for x in np.unique(classNames)]

    # 📦 Initialize image preprocessors
    aap = AspectAwarePreprocessor(64, 64)
    iap = ImageToArrayPreprocessor()

    # 📊 Load dataset from disk and scale raw pixel intensities to range [0, 1]
    sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
    (data, labels) = sdl.load(imagePaths, verbose=500)
    data = data.astype('float') / 255.0

    # 🎲 Split data into training and testing sets
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

    # 📈 Convert labels from integers to vectors
    trainY = LabelBinarizer().fit_transform(trainY)
    testY = LabelBinarizer().fit_transform(testY)

    # 🌟 Constructing image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode='nearest')

    # 🧪 Initializing the optimizer and model
    print('[INFO] Compiling model ...')
    opt = SGD(learning_rate=0.025)  # Updated
    model = MerguiNet.build(width=64, height=64, depth=3, classes=len(classNames))
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # 🚀 Train network
    print('[INFO] 🌟 Training network ...')
    H = model.fit(aug.flow(trainX, trainY, batch_size=32), validation_data=(testX, testY),
                            steps_per_epoch=len(trainX) // 32, epochs=100, verbose=1)

    # 📊 Evaluate the model
    print('[INFO] Evaluating the network ...')
    predictions = model.predict(testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))

    # 📝 Check the test loss and test accuracy
    score = model.evaluate(testX, testY, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1] * 100)

    # 🖼️ Serialize the network
    print("[INFO] Serializing network...")
    plot_model(model, to_file="merguinet.png", show_shapes=True)

    # 📈 Plot training loss and accuracy
    plt.style.use("classic")
    plt.figure()
    plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(args["output"])

if __name__ == "__main__":
    main()
