# Hot Encoding Data (previous commit) : https://towardsdatascience.com/learning-one-hot-encoding-in-python-the-easy-way-665010457ad9
# Training the data: https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/
# Cleaning and Accuracy : https://www.kaggle.com/rupamshil/loan-prediction-using-machine-learning

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing._data import StandardScaler

data = pd.read_csv('train_data.csv')

data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].mean()) # Fills empty loan data with the overall mean as a range of values are found
data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].median()) # Fills empty credit history with median values since it's only either 1 or 0

data.dropna(inplace=True) # Dropped the other non-major missing value, as it is hard to accuraately predict. Dropped data includes: Applicant Income, CoApplicant Income, LoanAmount, LoanAmountTerm

# Encoding is used as it is easier for the model to work with and make predictions off numerical data. ONe Hot Encoding was replaced as it is easier to implement without much loss in accuracy.
data['Loan_Status'].replace('Y',1,inplace = True)
data['Loan_Status'].replace('N',0,inplace = True)

data['Gender'].replace('Male', 1, inplace = True) 
data['Gender'].replace('Female', 0, inplace = True)

data['Married'].replace('Yes', 1, inplace = True) 
data['Married'].replace('No', 0, inplace = True)

data['Dependents'].replace('0', 0, inplace = True) 
data['Dependents'].replace('1', 1, inplace = True)
data['Dependents'].replace('2', 2, inplace = True)
data['Dependents'].replace('3+', 3, inplace = True) 

data['Education'].replace('Graduate', 1, inplace = True) 
data['Education'].replace('Not Graduate', 0, inplace = True)

data['Self_Employed'].replace('Yes', 1, inplace = True) 
data['Self_Employed'].replace('No', 0, inplace = True)

data['Property_Area'].replace('Rural', 0, inplace = True) 
data['Property_Area'].replace('Semiurban', 1, inplace = True)
data['Property_Area'].replace('Urban', 2, inplace = True) 


input_variables = data.drop(['Loan_ID', 'Loan_Status'], axis=1).values # Many input variables except for Loan ID
output_variable = data['Loan_Status'].values # Only one output variable - Loan Status

input_train, input_test, output_train, output_test = train_test_split(input_variables, output_variable ,test_size=0.2, random_state=0) # 80% of the data is used for training and 20% used for testing

# Standardizes the data to make it easier to compare
scaler = StandardScaler()
input_train = scaler.fit_transform(input_train)
input_test = scaler.transform(input_test)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=6, activation='relu')) # Input layer
model.add(tf.keras.layers.Dense(units=6, activation='relu')) # Hidden layer
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) # Output layer

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy']) # Binary Crossentropy is the most preferred loss function for binary classification problems. Adam is the preferred optimization algorithm. Accuracy metrics are stored when the model is being trained.

model.fit(input_train, output_train, batch_size =32, epochs =100) # Standard batch size and epochs

# model.save('loan_dataset_model.h5') - In case the model needs to be saved as .h5. Firebase accepts TFLite models for upload.

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('loan_dataset_model.tflite', 'wb') as f:
    f.write(tflite_model)

output_prediction = model.predict(input_test)
output_prediction = (output_prediction > 0.5) # Convert output prediction to 1 if >0.5 in the sigmoid function

print(accuracy_score(output_test, output_prediction)) # Accuracy Metrics collected when the model was trained, can be verified from a confusion matrix with the formula (Total Positive + Total Negative)/total data