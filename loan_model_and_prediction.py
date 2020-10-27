# Hot Encoding Data : https://towardsdatascience.com/learning-one-hot-encoding-in-python-the-easy-way-665010457ad9
# Training the data: https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/
# Prediction and Accuracy : https://www.kaggle.com/rupamshil/loan-prediction-using-machine-learning

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score

data = pd.read_csv('train_data.csv')

data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].mean()) # Fills empty loan data with the overall mean as a range of values are found
data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].median()) # Fills empty credit history with median values since it's only either 1 or -

data.dropna(inplace = True) # Dropped the other non-major missing value. inplace makes sure a new object copy is not created. Dropped data includes: Applicant Income, CoApplicant Income, LoanAmount, LoanAmountTerm

# One Hot Encoding is used so the data set is not biased. e.g, if Rural = 0, Urban = 1, Semiurban = 2, it will be biased towards Semiurban simply due to the higher count. This also removes the need for Feature Scaling

for col in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']:
    data[col] = data[col].astype('category')

data = pd.get_dummies(data=data,columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'])

# Manual 1 or 0 is only done for the loan status as it is the only Output variable, so One Hot Encoding does not need to be used
data['Loan_Status'].replace('Y', 1, inplace = True) 
data['Loan_Status'].replace('N', 0, inplace = True)

input_variables = data.iloc[:,1:-1].values # Many input variables except for Loan ID
output_variable = data.iloc[:,-1].values # Only one output variable - Loan Status

input_train,input_test, output_train, output_test = train_test_split(input_variables, output_variable, test_size=0.2, random_state=0) # 80% of the data is used for training and 20% used for testing


model = Sequential()
model.add(Dense(20, input_dim=20, activation='relu')) # Input Layer
model.add(Dense(20, input_dim= 20, activation='relu')) # Hidden Layer
model.add(Dense(1, activation='sigmoid')) # Output Layer
model.compile(loss = 'binary_crossentropy',optimizer='adam', metrics=['accuracy']) # Binary Crossentropy is the most preferred loss function for binary classification problems. Adam is the preferred optimization algorithm. Accuracy metrics are stored when the model is being trained. 

model.fit(input_train, output_train, batch_size=32, epochs=100) # Standard batch size and epochs
model.save('loan_dataset_model.h5')
output_prediction = model.predict(input_test).astype("int32")
output_prediction = (output_prediction > 0.5) # Convert output prediction to 1 if >0.5 in the sigmoid function

confusion_matrix = confusion_matrix(output_test, output_prediction) # Matrix made up of Predicted FALSE, Predicted TRUE, Actual FALSE and Actual True
print(confusion_matrix)
print(accuracy_score(output_test, output_prediction)) # Accuracy Metrics collected when the model was trained, can be verified from confusion matrix with the formula (Total Positive + Total Negative)/total data