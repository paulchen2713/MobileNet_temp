import numpy as np
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing import image
# tf.keras.preprocessing.image.load_img(image_path)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils

from IPython.display import Image
from sklearn.metrics import confusion_matrix

import itertools
import os
import shutil
import random
# import glob
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

"""
(noticed: the following codes activate only when using GPU to run the model, 
 but I'm not using GPU support I think? )
"""
# # check to be sure that TensorFlow is able to identify the GPU

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(physical_devices))

# # set_memory_growth() allocate only as much GPU memory as needed at a given time, 
# # and continues to allocate more when needed

# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# (IndexError                                Traceback (most recent call last)
# <ipython-input-3-df3199446273> in <module>
#      37 # set_memory_growth() allocate only as much GPU memory as needed at a given time,
#      38 # and continues to allocate more when needed
# ---> 39 tf.config.experimental.set_memory_growth(physical_devices[0], True)

# IndexError: list index out of range)

# downloading a copy of a single pretrained MobileNet,
# with weights that were saved from being trained on ImageNet images
mobile = tf.keras.applications.mobilenet.MobileNet()

# prepare_image() takes an file name, and processes the image to get it in a format that MobileNet expects
def prepare_image(file):
    img_path = 'D:/Deep_Learning_Projects/Emotion_Detection/'  # defining the relative path to the images
    # load_img(img_path) takes the image file, and resizing it to be of size (224, 224), 
    # and returns an instance of a PIL image
    img = tf.keras.preprocessing.image.load_img(img_path + file, target_size=(224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)    # then converting the PIL image into an array
    # input_array = np.array([input_array])                    # converting single image to a batch

    # numpy.expand_dims(a, axis) insert a new dimension in "axis th" dimension, and 
    # all previous dimensions would be push to the right
    img_array_dims_expand = np.expand_dims(img_array, axis=0)
    # preprocess_inout() scale the pixel values in the image between -1 to 1, same format as the images that 
    # MobileNet was originally trained on, and return the preprocessed image data as a numpy array
    return tf.keras.applications.mobilenet.preprocess_inout(img_array_dims_expand)

from Ipython.display import Image
Image(filename='', width=300, height=200)

processed_image = prepare_image('')
predictions = mobile.predict(precessed_image)

results = imagenet_utils.decode_predictions(predictions)

# organize data into train, valid, and test dirs, automatically
os.chdir('')
if os.path.isdir('train/0/') is False:
    os.mkdir('train')
    os.mkdir('valid')
    os.mkdir('test')

    for i in range(0, 7):
        shutil.move(f'{i}', 'train')
        os.mkdir(f'valid/{i}')
        os.mkdir(f'test/{i}')

        valid_samples = random.sample(os.listdir(f'train/{i}'), 30)
        for j in valid_samples:
            shutil.move(f'train/{i}/{j}', f'valid/{i}')

        test_samples = random.sample(os.listdir(f'train/{i}'), 5)
        for j in test_samples:
            shutil.move(f'train/{i}/{j}', f'test/{i}')

os.chdir('../..')

train_path = ''
valid_path = ''
test_path = ''

train_batches = ImageDataGenerator(processing_function=tf.keras.applications.mobilnet.processing_input) \
    .flow_from_directory(directory=train_path, target_size=(224, 224), batch_size=10)
valid_batches = ImageDataGenerator(processing_function=tf.keras.application.mobilenet.processing_input) \
    .flow_from_directiry(directory=valid_path, target_size=(224, 224), batch_size=10)
test_batches  = ImageDataGenerator(processing_function=tf.keras.application.mobilenet.processing_input) \
    .flow_from_directory(directory=test_path,  target_size=(224, 224), batch_size=10, shuffle=False)

mobile = tf.keras.applications.mobilenet.MobileNet()
mobile.summary()

x = mobile.layers[-6].output

# functional model
output_layer = Dense(uints=7, activation='softmax')(x)

model = Model(inputs=mobile.input, outputs=output_layer)

for layer in model.layers[:-23]:
    layer.trainable = False

model.summary()

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_batches, 
        step_per_epoch=len(train_batches), 
        validation_data=valid_batches,
        validation_steps=len(valid_batches),
        epoches=30,
        verbose=2
)

test_labels = test_batches.classes

predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)


test_batches.class_indices

# confusion matrix
cm = coonfusion_matrix(y_true=test_labels, y_pred=predictions.argmax(axis=1))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

test_batches.class_indices

cm_plot_labels = ['0', '1', '2', '3', '4', '5', '6', '7']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')




