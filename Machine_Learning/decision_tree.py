
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
from operator import itemgetter
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn.datasets import load_wine
from IPython.display import SVG
from graphviz import Source
from IPython.display import display


def predict(X_train, X_test, y_train, y_test, labels):
    #todo
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
    print(best_depth)
    DTL_model = DecisionTreeRegressor(max_depth=best_depth)
    DTL_model.fit(X_train, y_train)
    cv_predict = DTL_model.predict(X_test)

    print("final simple model error: ", mean_absolute_error(y_test, simple_predict))
    print("final cv model error: ", mean_absolute_error(y_test, cv_predict))

    graph = Source(tree.export_graphviz(DTL_model, out_file=None
                                        , feature_names=labels, class_names=['0', '1', '2']
                                        , filled=True))
    display(SVG(graph.pipe(format='svg')))


    