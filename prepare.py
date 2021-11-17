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
    
    # drop the outlier in gender ("other"), reset index
    df = df.drop(3116).reset_index().drop(columns='index')
    
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