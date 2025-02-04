# Now importing libraries for program
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sonar_data = pd.read_csv('Rock Vs Mine Prediction-IDE file/sonar.all-data.csv', header=None)
print(sonar_data.head())

# rows X col
print(sonar_data.shape)

#Stastical measures
print(sonar_data.describe())

print(sonar_data[60].value_counts())

# M -> Mine R -> Rock
print(sonar_data.groupby(60).mean())

# seperating data and labels
X= sonar_data.drop(columns=60, axis=1)
Y= sonar_data[60]
print(X)
print(Y)

#Training and Test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify= Y, random_state=1)
print(X.shape, X_train.shape, X_test.shape) 

# Model Training
model= LogisticRegression()
#Training the model with our trained data
model= model.fit(X_train, Y_train)

#Accuracy of trained data
X_train_prediction= model.predict(X_train)
training_data_accuracy= accuracy_score(X_train_prediction, Y_train)
print('Accuracy on trained data ',training_data_accuracy) 

# Accuracy of test data
X_test_prediction= model.predict(X_test)
test_data_accuracy= accuracy_score(X_test_prediction, Y_test)
print('Accuracy on trained data ',test_data_accuracy)

# Making a predictive system
# Input field for putting the required sonar data...
input_data=(0.0366,0.0421,0.0504,0.0250,0.0596,0.0252,0.0958,0.0991,0.1419,0.1847,0.2222,0.2648,0.2508,0.2291,0.1555,0.1863,0.2387,0.3345,0.5233,0.6684,0.7766,0.7928,0.7940,0.9129,0.9498,0.9835,1.0000,0.9471,0.8237,0.6252,0.4181,0.3209,0.2658,0.2196,0.1588,0.0561,0.0948,0.1700,0.1215,0.1282,0.0386,0.1329,0.2331,0.2468,0.1960,0.1985,0.1570,0.0921,0.0549,0.0194,0.0166,0.0132,0.0027,0.0022,0.0059,0.0016,0.0025,0.0017,0.0027,0.0027)
#changing the input_data into numpy array
input_data_as_numpy_array=np.asarray(input_data)
#reshaping the array for one instanse prediction
input_data_as_numpy_array=np.asarray(input_data)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
# Result
prediction= model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]=='R'):
  print('The object is a Rock')
else:
  print('The object is a Mine')

