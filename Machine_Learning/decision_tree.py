import sklearn

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
from operator import itemgetter
import os
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
import matplotlib.pyplot as plt
import pydotplus
from Machine_Learning import Plot_output



def predict(X_train, X_test, y_train, y_test, labels, pic_name, dir):
    simple_model = DecisionTreeRegressor(max_depth=10)
    simple_model.fit(X_train, y_train)
    simple_predict = simple_model.predict(X_test)

    depth = []
    for i in range(3, 20):
        DTL_model = DecisionTreeRegressor(max_depth=i)
        # Perform 7-fold cross validation
        #todo check why scores gives little numbers
        scores = cross_val_score(estimator=DTL_model, X=X_train, y=y_train, cv=7, n_jobs=4)
        depth.append((i, scores.mean()))
    best_depth = max(depth, key=itemgetter(1))[0]
    DTL_model = DecisionTreeRegressor(max_depth=best_depth)
    DTL_model.fit(X_train, y_train)
    cv_predict = DTL_model.predict(X_test)

    print("mean absolute error: ", mean_absolute_error(y_test, simple_predict))
    print("r2 error: ", sklearn.metrics.r2_score(y_test, simple_predict))

    print("mean absolute error(cv): ", mean_absolute_error(y_test, cv_predict))
    print("r2 error(cv): ", sklearn.metrics.r2_score(y_test, cv_predict))


    # dot_data = StringIO()
    # export_graphviz(DTL_model, out_file=dot_data,feature_names=labels,
    #                 filled=True, rounded=True,
    #                 special_characters=False)
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # if not os.getcwd().__contains__("Graphs & Photos"):
    #     os.chdir(os.getcwd()[:os.getcwd().index("Excel_files")] + "Graphs & Photos")
    #
    # graph.set_size('"45,20!"')
    #
    # graph.write_png('DT nodes new test.png')

    Plot_output.plot_graph(X_test, y_test, cv_predict, pic_name, dir)
    Plot_output.plot_DT_nodes(DTL_model, labels, dir)



