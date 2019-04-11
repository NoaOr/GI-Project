from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.externals.six import StringIO
import pydotplus
import seaborn as sns
import matplotlib.pyplot as plt
import os
from fastai.structured import draw_tree

from sklearn.tree import export_graphviz


def predict(X_train, X_test, y_train, y_test, lables, pic_name1, pic_name2):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=10, stop=12, num=10)]
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

    plot_predicts(rf_random, lables, X_test, y_test, cv_predictions, pic_name1, pic_name2)



def plot_predicts(model, lables, X_test, y_test, predict, pic_name1, pic_name2):

    error = mean_absolute_error(y_test, predict)

    print("final cv random forest model error: ", error)

    best_random = model.best_estimator_
    feature_imp = best_random.feature_importances_

    features_dict = dict(zip(lables, feature_imp))
    f = [(d, c) for d, c in zip(lables, feature_imp)]
    features_str = ""
    for a, b in f:
        features_str += a + ": " + str("%.4f" % b) + ", "
    features_str = features_str[:-2]

    indices = np.flip(np.argsort(feature_imp))

    plt.title('Random Forest - Feature Importance')
    sns.barplot(x=feature_imp[indices], y=lables, linewidth=2.5)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.gcf().set_size_inches(15, 9.3, forward=True)
    if not os.getcwd().__contains__("Graphs & Photos"):
        os.chdir(os.getcwd()[:os.getcwd().index("Excel_files")] + "Graphs & Photos")
    plt.savefig(pic_name1 + '.png')

    plt.close()

    plt.scatter(X_test['Carbohydrt_(g)'], y_test, color='blue', s=15)
    plt.scatter(X_test['Carbohydrt_(g)'], predict, color='red', s=10)

    plt.xticks(())
    plt.yticks(())

    plt.legend(('GI vlaue', 'predict GI value'),
               shadow=True, loc=(0.67, 0.85), handlelength=1.5, fontsize=10)

    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 16,
            }
    plt.title(pic_name2, fontdict=font)
    plt.xlabel('Model Error = ' +
               str(mean_absolute_error(y_test, predict)) + '\n', fontsize=8)

    if not os.getcwd().__contains__("Graphs & Photos"):
        os.chdir(os.getcwd()[:os.getcwd().index("Excel_files")] + "Graphs & Photos")
    plt.savefig(pic_name2 + '.png')

    draw_tree(m.estimators_[0], X_train, precision=3)
    plt.show()

    #
    # dot_data = StringIO()
    # export_graphviz(best_random, out_file=dot_data, feature_names=lables,
    #                 filled=True, rounded=True,
    #                 special_characters=False)
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # os.chdir(os.getcwd()[:os.getcwd().index("Excel_files")] + "Graphs & Photos")
    # graph.write_png('RF_best_estimator_tree.png')




