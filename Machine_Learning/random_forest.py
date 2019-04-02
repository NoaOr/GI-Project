from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pandas as pd


def predict(X_train, X_test, y_train, y_test, lables):
    # rf = RandomForestRegressor(random_state=42)
    #
    # print("parameters currently in use by rnadom forest: ")
    # print(rf.get_params())

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(X_train, y_train)
    cv_predictions = rf_random.predict(X_test)

    print("final cv random forest model error: ", mean_absolute_error(y_test, cv_predictions))

    best_random = rf_random.best_estimator_
    feature_imp = best_random.feature_importances_



    # feature_imp = pd.Series(rf_random.feature_importances_, index=lables).sort_values(ascending=False)

    print(feature_imp)


