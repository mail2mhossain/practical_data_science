# load dogs vs cats dataset, reshape and save to a new file
from os import listdir
import numpy as np
from datetime import datetime
from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from keras.models import Model

def process_data(dataset_home):
    subdirs = ['dogs/', 'cats/']
    photos, labels = list(), list()
    # enumerate files in the directory
    for subdir in subdirs:
        newdir = dataset_home + subdir
        for file in listdir(newdir):
            output = 0.0
            if subdir.startswith('cat'):
                output = 1.0
            # load an image from file
            image = load_img(newdir + file, target_size=(224, 224))
            # convert the image pixels to a numpy array
            image = img_to_array(image)
            # reshape data for the model
            image = np.expand_dims(image, axis=0)
            # prepare the image for the VGG model
            image = imagenet_utils.preprocess_input(image)
            # store
            photos.append(image)
            labels.append(output)
    return photos, labels

init_time = datetime.now()
dataset_home = '../../Data/kaggle_dogs_vs_cats/train/'
photos, labels = process_data(dataset_home)
fin_time = datetime.now()
print("Data Processing time : ", (fin_time-init_time))

init_time = datetime.now()
# load the VGG16 network
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3)) 
# pass the images through the network and use the outputs as
# our actual features
photos = np.vstack(photos)
features = model.predict(photos)

# reshape the features so that each image is represented by
# a flattened feature vector of the `MaxPooling2D` outputs
features = features.reshape((features.shape[0], 100352))
fin_time = datetime.now()
print("Feature Extraction time : ", (fin_time-init_time))

init_time = datetime.now()
# save the features
#np.save('resnet50_dogs_vs_cats_train_features.npy', features)
#np.save('resnet50_dogs_vs_cats_train_labels.npy', labels)
np.savez_compressed('resnet50_dogs_vs_cats_train_data.npz', features, labels)

fin_time = datetime.now()
print("Feature Saving to disk time : ", (fin_time-init_time))