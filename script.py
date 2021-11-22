from sklearn.naive_bayes import GaussianNB

# print out a bunch of stuff to say what the program does
print('hi')

# take user inputs for health information on the columns

'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
       'gender_Male', 'ever_married_Yes', 'work_type_Never_worked',
       'work_type_Private', 'work_type_Self-employed', 'work_type_children',
       'residence_type_Urban', 'smoking_status_formerly smoked',
       'smoking_status_never smoked', 'smoking_status_smokes'],
        
input('What is your age?') 
input('Do you have hypertension? Yes/No')
input('Do you have heart disease? Yes/No')
input('What is your glucose level?')
input('What is your BMI?')
input('Are you a male? Yes/No')
input('Have you ever been married? Yes/No')
input('Have you ever worked? Yes/No')
input('What sector do you work in? Private/Self-employed')
input('Is your residence rural or urban?')
input('What is your smoking status? formerly/never/currently')


df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df = prepare.prep_data(df).drop(columns=['age_range',])
df['stroke'] = df['stroke'].astype('int64')
# set list of columns to one-hot encode
col_list = ['gender','ever_married','work_type','residence_type','smoking_status']
# apply one-hot encoding using above list
df = pd.get_dummies(df, columns=col_list, drop_first=True)
# split using same random state as explore stage
trainvalidate, test = train_test_split(df, test_size=.2, random_state=777)
train, validate = train_test_split(trainvalidate, test_size=.25, random_state=777)
# isolate target
X_train, y_train = train.drop(columns='stroke'), train.stroke
X_validate, y_validate = validate.drop(columns='stroke'), validate.stroke
X_test, y_test = test.drop(columns='stroke'), test.stroke
                                   
# build, fit our best model inline (store model into cache)
# create naive bayes model
nb = GaussianNB(var_smoothing=0.00001).fit(X_train, y_train)
# make predictions in new column
y_train_predictions['nb_best_model'] = nb.predict(X_train)
y_test = pd.DataFrame(y_test).rename(columns={'stroke':'out_actuals'})
y_test['nb_best_model']= nb.predict(X_test)

# ingest data, prep, split, scale, encode, etc
# fit on ingested data

# take user's inputs as a new line in dataframe


# output predict_proba results
Xnew = [[...], [...]]
ynew = nb.predict_proba(Xnew)
                                   
print(f'you have a {} chance of stroke')