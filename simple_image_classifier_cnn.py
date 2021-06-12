# This is a sample Convolutional Neural Network (CNN) code used to determine two classes of image. We can use CNN for Binary Classifier,
# Multiclass Classifier or Continuous Regressor

# CNN has a number of steps that must be followed in order
# Step 1) Convolution of data. Convolution is a means of one function modifying another function. It can also be described as
# the integration of 2 functions. The data needs to go through multiple feature detector/ filter/ kernel to create multiple
# feature map that contains important signal from a frame of the image. A rectifier can be included after the convolution of the
# data to increase the non linearity of the feature map.
# Step 2) Pool the feature map and create a pooled data that contains compressed information based on the type of
# pooling that you have decided to use. For example, if you were to use max pooling, within the frame of the feature
# map, the max value will be selected while the others will be ignored. This process enables the CNN to keep only the
# useful/ important features of the image. Moreover, since the image is compressed, the size and the parameters will
# be smaller which enables faster processing.
# Step 3) Flattening. The pooled feature maps will be flattened into a single dimension (1D) vector which will then be fed
# into a full connection.
# Step 4) Full connection. The flattened data will be fed into an Aritifical Neural Network (ANN) framework and generate an output
# by minimizing the loss function via adjustments of the weights of the synapes/ connections using Gradient Descent.
# Step 5) Inclusion of an output layer. Similar to ANN, CNN will use backpropogation to adjust the weights. Loss function for
# Multi classification problem is usually Softmax, while for binary classification, it will be Binary Cross Entropy and lastly
# for continuous value, it is usually, mean_squared_error

# Importing the libraries
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.__version__

# Preprocessing the Training set
# The training data needs to be preprocessed to ensure that it does not make the model overfit on the training data. It needs to
# undergo some geometric transformation. By transforming the training dataset, it will reduce the variance of the model.
# Details of the different transformations are available here:
# https://keras.io/api/preprocessing/image/#image_dataset_from_directory-function

folder_path_with_training_data = ""
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
training_set = train_datagen.flow_from_directory(folder_path_with_training_data,
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

# Preprocessing the Test set
# The testing data needs to be transformed as well but unlike the training data, it should only be rescaled as these data should
# be unobserved. Note the input size (target_size) should be the same. The lower the target size the faster the training time.

folder_path_with_testing_data = ""
test_datagen = ImageDataGenerator(rescale=1./255)
validation_set = test_datagen.flow_from_directory(folder_path_with_testing_data,
                                                  target_size=(64, 64),
                                                  batch_size=32,
                                                  class_mode='binary')


# If you had noticed, there is no y_train or y_test that labels the dataset as what class it actually is. CNN does not requires labels
# and can be considered as a unsupervised learning modelling method.

def image_cnn(training_set, validation_set, epochs=25):
    cnn = tf.keras.models.Sequential()
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3,
                                   activation='relu', input_shape=[64, 64, 3]))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Conv2D(
        filters=32, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
    cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    cnn.compile(optimizer='adam', loss='binary_crossentropy',
                metrics=['accuracy'])
    cnn.fit(x=training_set, validation_data=validation_set, epochs=epochs)

    return cnn

cnn = image_cnn(training_set, validation_set, 25)

# Making a prediction
# If we were to to make a prediction for a new test image, we need to make sure that we use the appropriate transformation.
# The new image needs to be loaded using Keras. The size of the image needs to be transformed to the target size that we used
# for training. We need to then change the dimension of the image. Since we are feeding the data as a batch, the dimension of the
# new image needs to be expanded to that same size. 

new_image_path = ""

test_image = image.load_img(new_image_path, target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = cnn.predict(test_image)

# You will be able to know what class the image belongs to by typing the attribute class_indices to either the training or validation
# set
training_set.class_indices

if result[0][0] == 1:
    prediction = 'A'
else:
    prediction = 'B'

print(prediction)
