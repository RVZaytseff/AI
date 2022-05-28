#-------------------------------------------------------------------------------
# Name:        telecom_churn
# Purpose:
#
# Author:      rzaytsev
#
# Created:     27.05.2022
# Copyright:   (c) rzaytsev 2022
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import pandas as pd

# Building Model

def main():

    # save filepath to variable for easier access
    churn_file_path = 'telecom_churn.csv'

    # read the data and store data in DataFrame titled melbourne_data
    churn_data = pd.read_csv(churn_file_path)

    # print column list
    print(churn_data.columns)

    # print a summary of the data in Melbourne data
    print(churn_data.describe())

    # dropna drops missing values (think of na as "not available")
    churn_data = churn_data.dropna(axis=0)
    print(churn_data)

    # data for predict
    y = churn_data['Customer service calls']
    print('Customer service calls')
    print('----------------------')
    print(y)

    # column list for features
    churn_columns = ['Total day charge', 'Total night charge', 'Total eve charge', 'Total intl charge']
    X = churn_data[churn_columns]
    print(X)
    print(X.describe())

    # Building Model
    from sklearn.tree import DecisionTreeRegressor
    # Define model. Specify a number for random_state to ensure same results each run
    churn_model = DecisionTreeRegressor(random_state=1)

    # Fit model
    churn_model.fit(X, y)

    print("Making predictions for the following 5 items:")
    print(X.head())
    print("The predictions are")
    churn_predict_calls = churn_model.predict(X)
    print(churn_predict_calls)

    # calculate the mean absolute error
    from sklearn.metrics import mean_absolute_error
    churn_predict_error = mean_absolute_error(y, churn_predict_calls)
    print('churn_predict_error')
    print('-------------------')
    print(churn_predict_error)

    # Model Validation
    from sklearn.model_selection import train_test_split

    # split data into training and validation data, for both features and target
    # The split is based on a random number generator. Supplying a numeric value to
    # the random_state argument guarantees we get the same split every time we
    # run this script.
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
    # Define model
    churn_model = DecisionTreeRegressor()
    # Fit model
    churn_model.fit(train_X, train_y)

    # get predicted prices on validation data
    val_predictions = churn_model.predict(val_X)
    print('\n Decision Tree Regressor modelling')
    print('------------------------------------')
    print(mean_absolute_error(val_y, val_predictions))
    print('------------------------------------\n')


    ##5 Underfitting and Overfitting
    def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
        model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
        model.fit(train_X, train_y)
        preds_val = model.predict(val_X)
        mae = mean_absolute_error(val_y, preds_val)
        return(mae)

    # compare MAE with differing valuess of max_leaf_nodes
    for max_leaf_nodes in [5, 50, 500, 5000]:
        my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
        print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

    ##6 Random Forests
    # Using a more sophisticated machine learning algorithm.
    from sklearn.ensemble import RandomForestRegressor

    forest_model = RandomForestRegressor(random_state=1)
    forest_model.fit(train_X, train_y)
    churn_preds = forest_model.predict(val_X)
    print('\nRandom Forests modelling')
    print('------------------------')
    print(mean_absolute_error(val_y, churn_preds))
    print('------------------------\n')


    ##7 Missing Values
    from sklearn.ensemble import RandomForestRegressor
    X_train, X_valid, y_train, y_valid = train_X, val_X, train_y, val_y

    # Function for comparing different approaches
    def score_dataset(X_train, X_valid, y_train, y_valid):
        model = RandomForestRegressor(n_estimators=10, random_state=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        return mean_absolute_error(y_valid, preds)

    # Score from Approach 1 (Drop Columns with Missing Values)
    # Get names of columns with missing values
    cols_with_missing = [col for col in train_X.columns if train_X[col].isnull().any()]

    # Drop columns in training and validation data
    reduced_X_train = train_X.drop(cols_with_missing, axis=1)
    reduced_X_valid = val_X.drop(cols_with_missing, axis=1)

    print("MAE from Approach 1 (Drop columns with missing values):")
    print(score_dataset(reduced_X_train, reduced_X_valid, train_y, val_y))

    # Score from Approach 2 (Imputation)
    from sklearn.impute import SimpleImputer

    # Imputation
    my_imputer = SimpleImputer()
    imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
    imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

    # Imputation removed column names; put them back
    imputed_X_train.columns = X_train.columns
    imputed_X_valid.columns = X_valid.columns

    print("MAE from Approach 2 (Imputation):")
    print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))


    # Score from Approach 3 (An Extension to Imputation)
    # Make copy to avoid changing original data (when imputing)
    X_train_plus = X_train.copy()
    X_valid_plus = X_valid.copy()

    # Make new columns indicating what will be imputed
    for col in cols_with_missing:
        X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
        X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

    # Imputation
    my_imputer = SimpleImputer()
    imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
    imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

    # Imputation removed column names; put them back
    imputed_X_train_plus.columns = X_train_plus.columns
    imputed_X_valid_plus.columns = X_valid_plus.columns

    print("MAE from Approach 3 (An Extension to Imputation):")
    print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))


    ## Categorical Variables
    ## Three Approaches



    pass

if __name__ == '__main__':
    main()
