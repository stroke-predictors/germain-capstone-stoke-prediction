from sklearn.naive_bayes import GaussianNB

# print out a bunch of stuff to say what the program does
print('hi')

# take user inputs for health information on the columns
input('give me health data')

# build, fit our best model inline (store model into cache)
model = GaussianNB(var_smoothing=.00001)
# ingest data, prep, split, scale, encode, etc
# fit on ingested data

# take user's inputs as a new line in dataframe


# output predict_proba results
print('you have a 65% chance of stroke')