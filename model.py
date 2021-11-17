import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

from imblearn.combine import SMOTETomek 

# ------------------------- print results ------------------------- #

def print_classification_results(y_insample, y_outsample):
    """ Takes in a dataframe with column-separated model predictions, 
        Calculates accuracy and recall of each model (loops columns) for in- and out-sample data,
        Appends all results to running dataframe, 
        Returns dataframe. """
    # create empty results dataframe
    running_df = pd.DataFrame(columns=['Model','InSample_Accuracy','OutSample_Accuracy'])
    # loop through each model
    for model in y_insample.columns[1:]:
        # calculate model accuracy
        in_accuracy = (y_insample[model] == y_insample.in_actuals).mean()
        out_accuracy = (y_outsample[model] == y_outsample.out_actuals).mean()
        # determine sums of true positives and false negatives for recall calculation
            # true positive: model correctly predicts 1 when actual is 1
            # false negative: model wrongly predicts 0 when actual is 1
        in_true_positive = ((y_insample[model] == 1) & (y_insample['in_actuals'] == 1)).sum()
        in_false_negative = ((y_insample[model] == 0) & (y_insample['in_actuals'] == 1)).sum()
        out_true_positive = ((y_outsample[model] == 1) & (y_outsample['out_actuals'] == 1)).sum()
        out_false_negative = ((y_outsample[model] == 0) & (y_outsample['out_actuals'] == 1)).sum()
        # calculate recall scores
        in_recall = in_true_positive / (in_true_positive + in_false_negative)
        out_recall = out_true_positive / (out_true_positive + out_false_negative)
        # add results to new row in dataframe
        running_df = running_df.append({'Model':model,
                                        'InSample_Accuracy':in_accuracy, 
                                        'OutSample_Accuracy':out_accuracy,
                                        'InSample_Recall':in_recall,
                                        'OutSample_Recall':out_recall},
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

def manual_baseline(y_insample, y_outsample, baseline_value):
    """ Add a column for the manually-selected baseline prediction """
    # set each value to the chosen baseline value
    y_insample['manual_baseline'] = baseline_value
    y_outsample['manual_baseline'] = baseline_value

    return y_insample, y_outsample # return df with baseline predictions column

def classification_bl(y_insample, y_outsample):
    """ Calculate baseline using mode class for model comparison """
    # find baseline
    mode = y_insample.in_actuals.mode().tolist()[0]
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
        y_insample['tree_maxdepth' + str(depth)] = tree.predict(X_insample)
        y_outsample['tree_maxdepth' + str(depth)] = tree.predict(X_outsample)

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
    smooth_levels = [.000000001, .00000001, .0000001, .000001, .00001, .0001, .001, .01, 10, 100]
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

# ------------------------- pre-processing ------------------------- #

def Min_Max_Scaler(X_train, X_validate, X_test):
    """
    Takes in X_train, X_validate and X_test dfs with numeric values only
    Returns scaler, X_train_scaled, X_validate_scaled, X_test_scaled dfs
    """
    #Fit the thing
    scaler = MinMaxScaler().fit(X_train)
    #transform the thing
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), index = X_train.index, columns = X_train.columns)
    X_validate_scaled = pd.DataFrame(scaler.transform(X_validate), index = X_validate.index, columns = X_validate.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index = X_test.index, columns = X_test.columns)
    return scaler, X_train_scaled, X_validate_scaled, X_test_scaled

def smoter(X_train, y_train):
    """ Use SMOTE+Tomek to eliminate class imbalances for train split """
    # build SMOTE
    smtom = SMOTETomek(random_state=123)
    # SMOTE the train set
    X_train_smtom, y_train_smtom = smtom.fit_resample(X_train, y_train)
    # show before-and-after
    print("Before SMOTE applied:", X_train.shape, y_train.shape)
    print("After SMOTE applied:", X_train_smtom.shape, y_train_smtom.shape)

    return X_train_smtom, y_train_smtom # return SMOTE-d train data