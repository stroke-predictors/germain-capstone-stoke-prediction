import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import prepare


X_train, y_train, X_validate, y_validate, X_test, y_test = prepare.post_analysis_model_prep()

# call model
tree = DecisionTreeClassifier(max_depth=1, random_state=123).fit(X_train, y_train)

# print out a bunch of stuff to say what the program does
print('Hello! This program predicts your probility of stroke by asking you a few questions and then comparing your answers our best model.')

# take user inputs for health information on the columns

# Index(['avg_glucose_level', 
#        'age', 
#        'hypertension',
#        'heart_disease',
#        'ever_married_Yes']

# Take user input
age = input('What is your age? \n') 
glucose = input('What is your glucose level in mg/dL? range (0-300) \n')
ht = input('Do you have hypertension? yes/no \n')
hd = input('Do you have heart disease? yes/no \n')
married = input('Have you ever been married? yes/no \n')

# place user input in dictionary
user_inputdict = {'avg_glucose_level': glucose, 
                  'age': age, 
                  'hypertension_0': ht=='no', 
                  'hypertension_1': ht=='yes',
                  'heart_disease_0': hd=='no', 
                  'heart_disease_1': hd=='yes', 
                  'ever_married_No': married=='no',
                  'ever_married_Yes': married=='yes'
                 }

# grab column names
user_input = X_train.head(0)
# add user_inputdict values to the appropriate columns
user_input1 = user_input.append(user_inputdict, ignore_index=True)

# make predictions based on users input
pred = tree.predict_proba(user_input1)

#
print(f'You have a {round((pred[0][1]), 3)*100}% chance of stroke')