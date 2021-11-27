# This program calculates a user's risk of stroke based on different factors like age.
# The calculation is done by a predictive model with Recall: 83%, Accuracy: 65%, and ROC AUC: 85% on out-of-sample data.
# The model was created by the Stroke Prediction capstone team from Codeup's Germain data science cohort.
# The data to train the model comes from a healthcare dataset involving a person's age, BMI, glucose level, and more.
# The team members are: Randy French, Sarah Lawson Woods, Carolyn Davis, Elihizer Lopez, and Jacob Paxton.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from imblearn.combine import SMOTETomek 

def prep_data():
    """
        Ingests the healthcare dataset,
        Drops the same rows that were dropped for analysis,
        Cleans and encodes data as necessary,
        Limits the data to the required features,
        Splits the data in the same way as was done for the team's analysis,
        Isolates the target from the split needed to train the model,
        Oversamples the data the same way it was done for the analysis,
        Return the data needed to train the model.
    """
    # ingest data
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    # drops a few rows that were dropped for other reasons in analysis
    df = df.drop([3116,2128,4209]).reset_index().drop(columns='index')
    # create features
    df['stroke'] = df['stroke'] == 1
    df['high_glucose'] = df['avg_glucose_level'] >= 125
    df['has_hypertension'] = df['hypertension'] == 1
    df['has_heart_disease'] = df['heart_disease'] == 1
    df['ever_married'] = df['ever_married'] == 'Yes'
    # limit to required features
    df = df[['stroke','age','high_glucose','has_hypertension','has_heart_disease','ever_married']]
    # split data
    train_validate, test = train_test_split(df, test_size=.2, random_state=777)
    train, validate = train_test_split(train_validate, test_size=.25, random_state=777)
    # isolate target
    X_train, y_train = train.drop(columns='stroke'), train.stroke
    # SMOTE+Tomek oversampling
    """ Use SMOTE+Tomek to eliminate class imbalances for train split """
    # build SMOTE
    smtom = SMOTETomek(random_state=123)
    # SMOTE the train set
    X_train, y_train = smtom.fit_resample(X_train, y_train)
    # return data needed to train model
    return X_train, y_train

def calculate_risk(user_input_row):
    """
        Re-creates the best-performing model from the Stroke Prediction team's analysis,
        Fits it on the data used in the analysis,
        Use sklearn's predict_proba method to calculate the risk of stroke,
        Return the calculated number.
    """
    X_train, y_train = prep_data()
    model = GaussianNB(var_smoothing=.01).fit(X_train, y_train)
    calculated_risk = model.predict_proba(user_input_row)
    calculated_risk = int(calculated_risk[0][1] * 100)

    return calculated_risk

print('-'*40)
print('Hello! This program predicts stroke risk.')
print('Please answer the following questions:\n')

age = float(input('How old are you, in years?\n'))
glucose = int(input('What is your average glucose level?\n'))
hypertension = str(input('Do you have hypertension? y/n\n'))
heart_disease = str(input('Do you have heart disease? y/n\n'))
ever_married = str(input('Have you ever been married/are currently married? y/n\n'))

user_dict = {'age':[age],
             'high_glucose': [(glucose >= 125)],
             'has_hypertension': [(hypertension == 'y')],
             'has_heart_disease': [(heart_disease == 'y')],
             'ever_married': [(ever_married == 'y')],}

user_input_row = pd.DataFrame(user_dict)

print(f"\nOur model calculates your risk of stroke is: {calculate_risk(user_input_row)}%")

quit()