# vgg16 model used for transfer learning on the dogs and cats dataset
import sys
from matplotlib import pyplot
from sklearn.metrics import classification_report,confusion_matrix, plot_confusion_matrix
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation, Dropout, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator

# define cnn model
def define_model():
	# load model
	model = VGG16(include_top=False, input_shape=(224, 224, 3))
	# mark loaded layers as not trainable
	for layer in model.layers:
		layer.trainable = False
	# add new classifier layers
	flat1 = Flatten()(model.layers[-1].output)
	#class1 = Dense(256, activation='relu', kernel_initializer='he_uniform')(flat1)
	# output = Dense(1, activation='sigmoid')(class1)
	# Block #1: FC => RELU layers
	dense1 = Dense(4096, activation='relu', kernel_initializer='he_uniform')(flat1)
	batchNorm1 = BatchNormalization()(dense1)
	drop1 = Dropout(0.5)(batchNorm1)
	# Block #2: FC => RELU layers
	dense2 = Dense(4096, activation='relu', kernel_initializer='he_uniform')(drop1)
	batchNorm2 = BatchNormalization()(dense2)
	drop2 = Dropout(0.5)(batchNorm2)
	# sigmoid classifier
	output = Dense(1, activation='sigmoid')(drop2)
	# define new model
	model = Model(inputs=model.inputs, outputs=output)
	model.summary()
	# compile model
	opt = Adam(lr=0.001)
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
	train_datagen = ImageDataGenerator(rescale=1.0/255.0, rotation_range=30, width_shift_range=0.1,
                            height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	                        horizontal_flip=True, fill_mode="nearest")
	test_datagen = ImageDataGenerator(rescale=1.0/255.0)
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