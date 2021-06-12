# How to apply a 3 layer Artificial Neural Network
# 1) Install tensorflow2
# 2) Must change any categorical variables into 0 and 1
# 3) Split train and test data
# 4) Standardize or Normalize the training and testing data. Use the fit from the training data to 
# scale the test the data since the test is supposed to be new unobserved data. It must be applied to
# all the features whether it is a categorical or not.
# 5) Choose the layers and the number of neurons. Choose a compiler and train the model
# 6) Check the accuracy of the model


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score


# Lets assume that your datasource is a CSV, note you can change this depending on your datasource. For 
# the sake of this example, lets assume that column 2 is Sex and column 3 is Country. So we need to 
# encode these variables. Lastly, lets assume that the last column is the Dependendent variable
file = ""
df = pd.read_csv(file)

# The data needs to be in a 2D array or a matrix, which means it needs to be in [[]] brackets.
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Lets encode sex, column is 2 so the index will be 1
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])

# Lets encode Country, column is 3 so the index will be 2.
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Prior to scaling we must split the dataset
# You could also create a validation set by splitting the training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardizing the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Now we can build our model
# choosing your activation function 
# https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/

# Note you can create a loop depending on how many hidden layers you want


def simple_ann(X_train, y_train, problem_type='binary', input_neuron=6, batch_size=32, epochs=100):
    if problem_type == 'binary':
        ann = tf.keras.models.Sequential()
        ann.add(tf.keras.layers.Dense(units=input_neuron, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=input_neuron, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=input_neuron, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        ann.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
    elif problem_type == 'regressor':
		ann = tf.keras.models.Sequential()
		ann.add(tf.keras.layers.Dense(units=input_neuron, activation='relu'))
		ann.add(tf.keras.layers.Dense(units=input_neuron, activation='relu'))
		ann.add(tf.keras.layers.Dense(units=input_neuron, activation='relu'))
		ann.add(tf.keras.layers.Dense(units=1))
		ann.compile(optimizer='adam', loss='mean_squared_error')
		ann.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

    return ann


# Train the model and test the accuracy. accuracy_score is for binary classifier
# There are many other scoring method
# https://scikit-learn.org/stable/modules/model_evaluation.html 
ann = simple_ann('binary', 6, 32, 100, X_train, y_train)
y_pred = ann.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
