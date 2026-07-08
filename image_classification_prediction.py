import pickle
import numpy as np
import skimage
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from sklearn.base import BaseEstimator, TransformerMixin



##################################################
# Same transformers used during training
##################################################

class RGB2GrayTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.array([skimage.color.rgb2gray(img) for img in X])


class HogTransformer(BaseEstimator, TransformerMixin):

    def __init__(self,
                 orientations=9,
                 pixels_per_cell=(14,14),
                 cells_per_block=(2,2),
                 block_norm='L2-Hys'):

        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        features = []

        for image in X:

            fd = hog(image,
                     orientations=self.orientations,
                     pixels_per_cell=self.pixels_per_cell,
                     cells_per_block=self.cells_per_block,
                     block_norm=self.block_norm)

            features.append(fd)

        return np.array(features)


##################################################
# Load Saved Objects
##################################################

clf = pickle.load(open("image_classifier.pkl", "rb"))
scaler = pickle.load(open("models/hog_scaler.pkl", "rb"))
labels = pickle.load(open("models/class_labels.pkl", "rb"))


##################################################
# Read Image
##################################################

image_path = input("Enter Path: ")

img = imread(image_path)

img = resize(img, (80,80))

img = np.array([img])


##################################################
# Feature Extraction
##################################################

grayify = RGB2GrayTransformer()
hogify = HogTransformer()

gray = grayify.transform(img)

hog_features = hogify.transform(gray)

hog_features = scaler.transform(hog_features)


##################################################
# Prediction
##################################################

prediction = clf.predict(hog_features)

print("Predicted Class :", prediction[0])