import pandas as pd
from sklearn.model_selection import train_test_split

import model

# ------------------------- prep for exploration ------------------------- #

def prep_data(df): # clean dataframe, impute nulls in BMI (exploration-ready)
    """ 
        Takes the original Kaggle dataset,
        Drops a row for an outlier in gender and reset index, 
        Drop the id column because the index serves the same,
        Converts ordinal columns to objects for one-hot encoding,
        Creates age_range feature for 5-year increments,
        Imputes BMI nulls using average BMI for observation's age range and gender, and
        Returns the prepared dataframe. This does not do model prep work.
    """
    
    # drop the outlier in gender ("other") and the outlier in BMI (97.6%), reset index
    df = df.drop([3116,2128]).reset_index().drop(columns='index')
    
    # drop id column (index is just as valuable)
    df = df.drop(columns='id')

    # fix the annoying capitalization
    df = df.rename(columns={'Residence_type':'residence_type'})
    
    # convert ordinal columns to objects
    df['hypertension'] = df.hypertension.astype('object')
    df['heart_disease'] = df.heart_disease.astype('object')

    # make age groups list
    five_year_cutpoints = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
    
    # create new column for age range
    df['age_range'] = pd.cut(x=df.age, bins=five_year_cutpoints).astype('string')
    
    # fix values in age range for something more readable (replaces ", ", "-" e.g. '0-5')
    df['age_range'] = df['age_range'].str[1:-1].str.replace(', ', '-').astype('object')

    # calculate mean BMI for each age range and gender combination
    grouped = df.groupby(['age_range', 'gender']).bmi.mean().round(1)
    
    # assign BMI to nulls using the average BMI for the observation's age range and gender
    # based on index for rows with null values for bmi
    df.loc[df.bmi.isna(), 'bmi'] = df[df.bmi.isna()].apply(lambda x: grouped[x.age_range][x.gender], axis=1)

    return df # return exploration-ready dataframe

def engineer_features(df):
    """ 
        One-hot encodes work_type, smoking_status, residence_type, gender, ever_married, and stroke columns,
        Converts ordinal 1s and 0s into True and False values for hypertension and heart_disease columns,
        Creates bmi_range column for BMI ranges (bins are width: 10%),
        Creates high_glucose column for avg_glucose_level above 125,
        Creates is_senior column for ages over 55,
        Drops original categorical columns
    """
    # print original shape
    print(f'Original shape: {df.shape[0]} rows, {df.shape[1]} columns.')
    # encoding: work_type
    df['govt_job'] = df['work_type'] == 'Govt_job'
    df['self_employed'] = df['work_type'] == 'Self-employed'
    df['private_work'] = df['work_type'] == 'Private'
    df['never_worked'] = (df['work_type'] == 'children') | (df['work_type'] == 'Never_worked')
    # encoding: smoking_status
    df['current_smoker'] = df['smoking_status'] == 'smokes'
    df['prior_smoker'] = df['smoking_status'] == 'formerly smoked'
    df['never_smoked'] = df['smoking_status'] == 'never smoked'
    # encoding: residence_type
    df['is_urban'] = df['residence_type'] == 'Urban'
    # encoding: gender
    df['is_female'] = df['gender'] == 'Female'
    # encoding: ever-married
    df['ever_married'] = df['ever_married'] == 'Yes'
    # encoding: stroke
    df['stroke'] = df['stroke'] == 1

    # convert ordinal to boolean: hypertension
    df['has_hypertension'] = df['hypertension'] == 1
    # convert ordinal to boolean: heart_disease
    df['has_heart_disease'] = df['heart_disease'] == 1

    # binning: bmi
    bmi_bins = [0,10,20,30,40,50,60,70,80,90,100]
    bmi_labels = ['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90-99']
    df['bmi_range'] = pd.cut(df.bmi, bins=bmi_bins, labels=bmi_labels)

    # glucose categorization
    df['high_glucose'] = df['avg_glucose_level'] >= 125
    # age categorization
    df['is_senior'] = df['age'] >= 55

    # drop old categorical columns
    df.drop(columns=['work_type','smoking_status','hypertension',
                     'heart_disease','residence_type', 'gender'], inplace=True)
    
    # order columns for nicer output
    col_list = ['stroke','age','age_range','is_senior','bmi','bmi_range',
                'avg_glucose_level','high_glucose',
                'has_hypertension', 'has_heart_disease',
                'ever_married','is_female','is_urban',
                'current_smoker','prior_smoker','never_smoked',
                'govt_job','self_employed','private_work','never_worked']
    df = df[col_list]

    # print new shape
    print(f'New shape: {df.shape[0]} rows, {df.shape[1]} columns.')

    # return engineered dataframe
    return df

# ------------------------- additional prep for modeling ------------------------- #

def model_prep(df): # encode, split, isolate target, scale, SMOTE cleaned kaggle dataset (model-ready)
    """ 
        Takes the dataframe already put through prep_data as input,
        One-hot encodes categorical and ordinal columns,
        Splits data into 60-20-20 train-validate-test splits,
        Isolates the target column from each split,
        Scales each split's features,
        Uses SMOTE to address class imbalance for train, and
        Return all prepared dataframes. Requires prior prep function.
    """
    # set list of columns to one-hot encode
    col_list = ['gender','ever_married','work_type','residence_type','smoking_status', 'age_range']
    
    # apply one-hot encoding using above list
    df = pd.get_dummies(df, columns=col_list, drop_first=True)
    
    # split
    trainvalidate, test = train_test_split(df, test_size=.2, random_state=777)
    train, validate = train_test_split(trainvalidate, test_size=.25, random_state=777)
    
    # isolate target for each split
    X_train, y_train = train.drop(columns='stroke'), train.stroke
    X_validate, y_validate = validate.drop(columns='stroke'), validate.stroke
    X_test, y_test = test.drop(columns='stroke'), test.stroke
    
    # apply scaling using the Min_Max_Scaler function from model.py
    scaler,\
    X_train_scaled,\
    X_validate_scaled,\
    X_test_scaled = model.Min_Max_Scaler(X_train, X_validate, X_test)
    
    # use SMOTE+Tomek to address class imbalances between stroke and not-stroke (function in model.py)
    X_train_smtom, y_train_smtom = model.smoter(X_train_scaled, y_train)

    # return dataframes required for modeling
    return X_train_smtom, y_train_smtom, X_validate_scaled, y_validate, X_test_scaled, y_test