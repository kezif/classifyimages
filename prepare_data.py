import cv2     # for capturing videos
import pandas as pd
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils
from keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split

data = pd.read_csv('mapping.csv')     # reading the csv file
print(data.head())     # printing first five rows of the file

X = [ ]     # creating an empty array
for img_name in data.Image_ID:
    img = cv2.imread(os.path.join('frames', img_name)).astype(int)
    X.append(img)  # storing each image in array X
    X = np.array(X)    # converting list to array

y = data.Class
dummy_y = np_utils.to_categorical(y)    # one hot encoding Classes

image = []
for i in range(0,X.shape[0]):
    a = cv2.resize(X[i], (224,224), interpolation=cv2.INTER_AREA).astype(int)      # reshaping to 224*224*3
    image.append(a)
X = np.array(image)


X = preprocess_input(X, mode='tf')      # preprocessing the input data
SPLITED_DATA = train_test_split(X, dummy_y, test_size=0.3, random_state=42)    # preparing the validation set


def prepare_image(path):
    img = cv2.imread(os.path.join('frames', img_name)).astype(int)
    X = np.array(img)
