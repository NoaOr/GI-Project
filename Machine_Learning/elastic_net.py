import sklearn
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error
from Machine_Learning import Plot_output


def predict(X_train, X_test, y_train, y_test, features, pic_name, dir):
    model = ElasticNetCV(cv=4)
    model.fit(X_train, y_train)

    predict = model.predict(X_test)
    print("mean absolute error: ", mean_absolute_error(y_test, predict))
    print("r2 error: ", sklearn.metrics.r2_score(y_test, predict))
    print("alpha: " , model.alpha_)
    print("alphas: ", model.alphas_)
    print("iter: ", model.n_iter_)

    x = len(features)
    y = len(model.coef_)
    coefficients = [(d, c) for d, c in zip(features, model.coef_)]
    coefficients_str = ""
    for a, b in coefficients:
        coefficients_str += a + ": " + str("%.4f" % b) + "\n"
    coefficients_str = coefficients_str[:-2]

    print("coef: ", coefficients_str)

    Plot_output.plot_coefficients(coefficients_str, pic_name=pic_name, dir=dir)
    Plot_output.plot_graph(X_test, y_test, predict, pic_name, dir)
