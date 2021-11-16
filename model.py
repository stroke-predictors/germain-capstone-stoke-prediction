import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# ------------------------- print results ------------------------- #

def print_nlp_results(y_insample, y_outsample):
    """ Takes in a dataframe with column-separated model predictions, 
        Calculates accuracy of each model (loops columns) for in- and out-sample data,
        Appends all results to running dataframe, 
        Returns dataframe. """
    # create empty results dataframe
    running_df = pd.DataFrame(columns=['Model','InSample_Accuracy','OutSample_Accuracy'])
    # loop through each model
    for model in y_insample.columns[1:]:
        # calculate model accuracy
        in_accuracy = (y_insample[model] == y_insample.in_actuals).mean()
        out_accuracy = (y_outsample[model] == y_outsample.out_actuals).mean()
        # add results to new row in dataframe
        running_df = running_df.append({'Model':model,
                                        'InSample_Accuracy':in_accuracy, 
                                        'OutSample_Accuracy':out_accuracy},
                                         ignore_index=True)

    return running_df # return results dataframe

# -------------------------- the shotgun -------------------------- #

def classification_shotgun(X_insample, y_insample, X_outsample, y_outsample):
    """ Take in Pandas Series for classification and target (pass columns!),
        Create several DecisionTree, RandomForest, LogisticRegression, Naive Bayes,
        and KNearest classification models, 
        Push model predictions to originating dataframe, return dataframe """
    # convert predictions column (usually Series) to dataframe
    if type(y_insample) != 'pandas.core.frame.DataFrame':
        y_insample = pd.DataFrame(y_insample.rename('in_actuals'))
    if type(y_outsample) != 'pandas.core.frame.DataFrame':
        y_outsample = pd.DataFrame(y_outsample.rename('out_actuals'))
    # Baseline - add predictions to df
    y_insample, y_outsample = classification_bl(y_insample, y_outsample)
    # Decision Tree classifier - add predictions to df
    y_insample, y_outsample = decisiontree(X_insample, y_insample, X_outsample, y_outsample)
    # Random Forest classifier - add predictions to df
    y_insample, y_outsample = randomforest(X_insample, y_insample, X_outsample, y_outsample)
    # Logistic Regression classifier - add predictions to df
    y_insample, y_outsample = logisticregression(X_insample, y_insample, X_outsample, y_outsample)
    # Naive Bayes classifier - add predictions to df
    y_insample, y_outsample = naivebayes(X_insample, y_insample, X_outsample, y_outsample)
    # K-Nearest Neighbors classifier - add predictions to df
    y_insample, y_outsample = knearestneighbors(X_insample, y_insample, X_outsample, y_outsample)
    
    return y_insample, y_outsample # return dataframes of predictions

# -------------------------- the models -------------------------- #

def classification_bl(y_insample, y_outsample):
    """ Calculate baseline using mode class for model comparison """
    # find baseline
    mode = y_insample.in_actuals.mode().item()
    # set baseline as prediction
    y_insample['baseline'] = mode
    y_outsample['baseline'] = mode

    return y_insample, y_outsample # return df with baseline predictions column

def decisiontree(X_insample, y_insample, X_outsample, y_outsample):
    """ Creates decision trees with max_depth 1,2,3,5,10 and random_state=123 """
    # set loop list
    max_depths = [1,2,3,5,10]
    # loop through max depths
    for depth in max_depths:
        # create decision trees
        tree = DecisionTreeClassifier(max_depth=depth, random_state=123)\
            .fit(X_insample, y_insample.in_actuals)
        # make predictions in new columns
        y_insample['cv_tree_maxdepth' + str(depth)] = tree.predict(X_insample)
        y_outsample['cv_tree_maxdepth' + str(depth)] = tree.predict(X_outsample)

    return y_insample, y_outsample # return dataframe with predictions appended

def randomforest(X_insample, y_insample, X_outsample, y_outsample):
    """ Creates random forests with max_depth 1,2,3,5,10 and random_state=123 """
    # set loop list
    max_depths = [1,2,3,5,10]
    # loop through max depths
    for depth in max_depths:
        # create random forest model
        rf = RandomForestClassifier(max_depth=depth, random_state=123)\
            .fit(X_insample, y_insample.in_actuals)
        # make predictions in new columns
        y_insample['rf_depth' + str(depth)] = rf.predict(X_insample)
        y_outsample['rf_depth' + str(depth)] = rf.predict(X_outsample)
    
    return y_insample, y_outsample # return dataframe with predictions appended

def logisticregression(X_insample, y_insample, X_outsample, y_outsample):
    """ Creates logistic regressions with random_state=123 """
    # create logistic regression model
    logit = LogisticRegression(random_state=123)\
        .fit(X_insample, y_insample.in_actuals)
    # add columns for predictions
    y_insample['logit'] = logit.predict(X_insample)
    y_outsample['logit'] = logit.predict(X_outsample)
    
    return y_insample, y_outsample # return dataframe with predictions appended

def naivebayes(X_insample, y_insample, X_outsample, y_outsample):
    """ Creates Naive-Bayes with var_smoothing of .001, .01, 10, 100 """
    # set loop list
    smooth_levels = [.001, .01, 10, 100]
    # loop through smoothing levels
    for smooth_level in smooth_levels:
        # create naive bayes model
        nb = GaussianNB(var_smoothing=smooth_level)\
            .fit(X_insample, y_insample.in_actuals)
        # make predictions in new column
        y_insample['nb_vsmooth' + str(smooth_level)] = nb.predict(X_insample)
        y_outsample['nb_vsmooth' + str(smooth_level)] = nb.predict(X_outsample)
    
    return y_insample, y_outsample # return dataframe with preds appended

def knearestneighbors(X_insample, y_insample, X_outsample, y_outsample):
    """ Create KNNs with neighbor counts of 3, 5, 10, 25, 75 """
    # set loop list
    neighbor_counts = [3,5,10,25,75]
    # loop through neighbor counts
    for neighbor_count in neighbor_counts:
        # create knn model
        knn = KNeighborsClassifier(n_neighbors=neighbor_count)\
            .fit(X_insample, y_insample.in_actuals)
        # make predictions in new column
        y_insample['knn_n' + str(neighbor_count)] = knn.predict(X_insample)
        y_outsample['knn_n' + str(neighbor_count)] = knn.predict(X_outsample)
    
    return y_insample, y_outsample # return dataframe with preds appended