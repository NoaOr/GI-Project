import pandas as pd
import os
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut, KFold
from scipy.stats import sem


def kf_cv(X_train, y_train, model):
    scores = np.zeros(X_train[:].shape[0])

    kf = KFold(n_splits=2, shuffle=False)  # Define the split - into 2 folds
    kf.get_n_splits(X_train)  # returns the number of splitting iterations in the cross-validator
    print(kf)
    KFold(n_splits=2, random_state=None, shuffle=False)

    for train_index, test_index in kf.split(X_train):
        X_train_cv, X_test_cv = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
        y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
        model = model.fit(X_train_cv, y_train_cv)
        y_pred = model.predict(X_test_cv)
        scores[test_index]= mean_absolute_error(y_test_cv.astype(int), y_pred.astype(int))
    print(("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores)))
    return model

# TODO: put this code in another class and return ml_df
if __name__ == '__main__':
    os.chdir(os.getcwd()[:os.getcwd().index("Machine_Learning")] + "Excel_files")
    df = pd.read_excel("GI_USDA_clean.xlsx")

    ml_df = df.drop(['CSFII 1994-96 Food Code', 'Food Description in 1994-96 CSFII',
                     'source table', 'reference food & time period', 'serve Size g',
                     'available cerbo hydrate', 'GL per serve', 'GI_2', 'acc', 'match-sent',
                     'GmWt_Desc2', 'GmWt_Desc1', 'Manganese_(mg)',
                     'GmWt_1', 'GmWt_2', 'Panto_Acid_mg)'], axis='columns')

    ml_df = ml_df.fillna(0)
    X = ml_df.drop('GI Value', axis=1, inplace=False)
    y = ml_df['GI Value'].values

    writer = pd.ExcelWriter('ML_table.xlsx', engine='xlsxwriter')
    ml_df.to_excel(writer, sheet_name='Sheet1')
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()

    DTL_model = DecisionTreeRegressor(max_depth=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

    # model1 = loo_cv(X_train, y_train, regr_1)
    cv_model = kf_cv(X_train, y_train, DTL_model)

    fit_model = DecisionTreeRegressor(max_depth=10)
    fit_model.fit(X_train, y_train)
    # Predict
    cv_predict = cv_model.predict(X_test)
    fit_predict = fit_model.predict(X_test)

    print("final cv model error: ", mean_absolute_error(y_test, cv_predict))
    print("final fit model error: ", mean_absolute_error(y_test, fit_predict))
