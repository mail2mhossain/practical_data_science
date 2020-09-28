# vgg16 model used for transfer learning on the dogs and cats dataset
import sys
from matplotlib import pyplot
from sklearn.metrics import classification_report,confusion_matrix, plot_confusion_matrix
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import Flatten, LeakyReLU
from keras.layers import Activation, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator

# function for creating a vgg block
def vgg_block(layer_in, n_filters, n_conv):
	# add convolutional layers
	for _ in range(n_conv):
		layer_in = Conv2D(n_filters, (3,3), padding="same", activation=LeakyReLU(), kernel_initializer='he_uniform')(layer_in)
		layer_in = BatchNormalization()(layer_in)
	
	# add max pooling layer
	layer_in = MaxPooling2D((2,2), strides=(2,2))(layer_in)
	layer_in = Dropout(0.25)(layer_in)
	return layer_in

# define cnn model
def define_model():
	# load model
	visible = Input(shape=(256, 256, 3))
	# Block #1: (CONV => RELU) * 2 => POOL
	layer = vgg_block(visible, 64, 2)
	# Block #2: (CONV => RELU) * 2 => POOL
	layer = vgg_block(layer, 128, 2)
	# Block #3: (CONV => RELU) * 3 => POOL
	layer = vgg_block(layer, 256, 3)
	# Block #4: (CONV => RELU) * 3 => POOL
	layer = vgg_block(layer, 512, 3)
	# Block #5: (CONV => RELU) * 3 => POOL
	layer = vgg_block(layer, 512, 3)

	flatten = Flatten()(layer)
	
	# Block #6: FC => RELU layers
	dense1 = Dense(4096, activation=LeakyReLU(), kernel_initializer='he_uniform')(flatten)
	batchNorm1 = BatchNormalization()(dense1)
	drop1 = Dropout(0.5)(batchNorm1)
	# Block #7: FC => RELU layers
	dense2 = Dense(4096, activation=LeakyReLU(), kernel_initializer='he_uniform')(drop1)
	batchNorm2 = BatchNormalization()(dense2)
	drop2 = Dropout(0.5)(batchNorm2)
	# sigmoid classifier
	output = Dense(1, activation='sigmoid')(drop2)
	# define new model
	model = Model(inputs=visible, outputs=output)
	
	# summarize model 
	model.summary()
	# compile model
	opt = Adam()
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

# run the test harness for evaluating a model
def run_test_harness():
	# define model
	model = define_model()
	# create data generator
	train_datagen = ImageDataGenerator(rescale=1.0/255.0, rotation_range=30, 
							width_shift_range=0.1,
                            height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	                        horizontal_flip=True, fill_mode="nearest")
	test_datagen = ImageDataGenerator(rescale=1.0/255.0)
	# specify imagenet mean values for centering
	#datagen.mean = [123.68, 116.779, 103.939]
	# prepare iterator
	train_it = train_datagen.flow_from_directory('../../Data/kaggle_dogs_vs_cats/train/',
		class_mode='binary', batch_size=64, target_size=(224, 224))
	test_it = test_datagen.flow_from_directory('../../Data/kaggle_dogs_vs_cats/test/',
		class_mode='binary', batch_size=64, target_size=(224, 224))
	
	# fit model
	history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
		validation_data=test_it, validation_steps=len(test_it), epochs=25, verbose=1)
	# evaluate model
	_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
	print('> %.3f' % (acc * 100.0))
	# learning curves
	#summarize_diagnostics(history)
    #predictions = model.predict_generator(test_it, b)
    #print(classification_report(testY.argmax(axis=1),
                                #predictions.argmax(axis=1), target_names=classNames))

# entry point, run the test harness
run_test_harness()
#define_model()