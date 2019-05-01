import sklearn

from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import os

def predict(X_train, X_test, y_train, y_test, features, pic_name):
    model = ElasticNetCV(cv=5, random_state=0)
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
        coefficients_str += a + ": " + str("%.4f" % b) + ", "
    coefficients_str = coefficients_str[:-2]
    index = coefficients_str.index("Phosphorus_(mg)")
    coefficients_str = coefficients_str[:index - 1] + '\n' + coefficients_str[index :]

    index = coefficients_str.index("Vit_B6_(mg)")
    coefficients_str = coefficients_str[:index - 1] + '\n' + coefficients_str[index:]

    index = coefficients_str.index("Beta_Carot_(µg)")
    coefficients_str = coefficients_str[:index - 1] + '\n' + coefficients_str[index:]

    print("coef: ", coefficients_str)

    plot_predict(X_test, y_test, predict, coefficients_str, pic_name)


def plot_predict(X_test, y_test, predict, coefficients_str, pic_name):
    plt.figure(figsize=(17, 12))

    plt.scatter(X_test['Carbohydrt_(g)'], y_test, color='blue', s=40)
    plt.scatter(X_test['Carbohydrt_(g)'], predict, color='red', s=35)

    plt.xticks(())
    plt.yticks(())

    plt.legend(('GI vlaue', 'predict GI value'),
               shadow=True, loc=(0.75, 0.85), handlelength=1.5, fontsize=20)

    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 30,
            }
    plt.title(pic_name, fontdict=font)
    plt.xlabel('Model Error = ' + str(mean_absolute_error(y_test, predict)) + '\n' +
               'R2 score = ' + str(sklearn.metrics.r2_score(y_test, predict)) +
               "coefficients: " + '\n' +
               coefficients_str, fontsize=10)

    if not os.getcwd().__contains__("Graphs & Photos"):
        os.chdir(os.getcwd()[:os.getcwd().index("Excel_files")] + "Graphs & Photos")
    plt.savefig(pic_name + '.png')
