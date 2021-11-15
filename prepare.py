import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def prep_data(df): # takes in dataFrame object
    df = df.drop(3116).reset_index().drop(columns='index') # index 3116 has gender 'Other', only one value. Drops value

    df['id'] = df.id.astype('object') # sets id as datatype 'object' instead of 'int'
    df['hypertension'] = df.hypertension.astype('object') # sets as datatype 'object' instead of 'int'
    df['heart_disease'] = df.heart_disease.astype('object') # sets as datatype 'object' instead of 'int'
    df['stroke'] = df.stroke.astype('object') # sets as datatype 'object' instead of 'int'

    five_year_cutpoints = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100] # creates list cut points
    df['age_range'] = pd.cut(x=df.age, bins=five_year_cutpoints).astype('string') # new column 'age_range' created, cuts 'age' by bins from 'five_year_cutpoints'

    # replaces ', ' with '-' for the values in age_range
    df['age_range'] = df['age_range'].str[1:-1]\ 
                    .str.replace(', ', '-')\
                    .astype('object')

    # object holding mean bmi values for each unique combination of 'age_range' and 'gender'
    grouped = df.groupby(['age_range', 'gender']).bmi.mean()

    # looks at index of rows where bmi holds a NaN value for column 'bmi'
    # applies lambda functions that takes in a row's index 
    # and replaces  the NaN with the appropriate bmi mean value based on 'age_range' and 'gender'
    df.loc[df.bmi.isna(), 'bmi'] = df[df.bmi.isna()].apply(lambda x: grouped[x.age_range][x.gender], axis=1)

    # renames column
    df.rename(columns={'Residence_type':'residence_type'}, inplace=True)
    
    # creates dummies using columns from df
    modeling_df = pd.get_dummies(df, drop_first=True)

    return df, modeling_df

    