import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import mean_absolute_error
import os
import matplotlib.patheffects as path_effects

font = {'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 40,
        }


def plot_graph(X_test, y_test, predict, pic_name, coefficients_str=""):
    print('Carbohydrt' + '\n\n' +
               'MAE = ' + str("%.3f" % mean_absolute_error(y_test, predict)) +
               ' MSE = ' + str(sklearn.metrics.mean_squared_error(y_test, predict)) +
               ' R2 = ' + str("%.3f" % sklearn.metrics.r2_score(y_test, predict)))


    plt.clf()
    plt.figure(figsize=(20, 13))

    plt.title(pic_name, fontdict=font)

    # plt.scatter(y_test, predict, color='blue', s=40)
    plt.scatter(X_test['Carbohydrt_(g)'], y_test, color='blue', s = 75)
    plt.scatter(X_test['Carbohydrt_(g)'], predict, color='red', s = 75)

    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)

    plt.legend(('GI vlaue', 'predict GI value'),
               shadow=True, loc=(0.7, 0.85), handlelength=1.5, fontsize=25)

    if coefficients_str == "":
        plt.xlabel('Carbohydrt' + '\n\n' +
                   'MAE = ' + str("%.3f" % mean_absolute_error(y_test, predict)) +
                   ' MSE = ' + str(sklearn.metrics.mean_squared_error(y_test, predict)) +
                    ' R2 = ' + str("%.3f" % sklearn.metrics.r2_score(y_test, predict)),
                    fontsize = 20)
    else:
        plt.xlabel('Carbohydrt' + '\n\n' +
                   'MAE = ' + str("%.3f" % mean_absolute_error(y_test, predict)) +
                   ' MSE = ' + str("%.3f" % sklearn.metrics.mean_squared_error(y_test, predict)) +
                   ' R2 = ' + str("%.3f" % sklearn.metrics.r2_score(y_test, predict)) +
                   " coefficients: " + coefficients_str,
                    fontsize = 20)

    plt.ylabel('GI value', fontsize = 20)

    if not os.getcwd().__contains__("Graphs & Photos"):
        os.chdir(os.getcwd()[:os.getcwd().index("Excel_files")] + "Graphs & Photos")
    plt.savefig(pic_name + '.png')

def plot_coefficients(coefficients_str, pic_name):
    plt.clf()

    fig = plt.figure(figsize=(5, 12))

    text = fig.text(0.5, 0.5, coefficients_str,
                    ha='center', va='center', size=15)
    text.set_path_effects([path_effects.Normal()])

    if not os.getcwd().__contains__("Graphs & Photos"):
        os.chdir(os.getcwd()[:os.getcwd().index("Excel_files")] + "Graphs & Photos")
    pic_name = "coefficients_" + pic_name
    plt.savefig(pic_name + '.png')

def plot_two_cols(x, y, df, pic_name):
    plt.clf()
    plt.figure(figsize=(20, 13))

    plt.title(pic_name, fontdict=font)

    plt.scatter(df[x], df[y], color='blue', s=40)

    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)

    plt.xlabel(x, fontsize=20)
    plt.ylabel(y, fontsize=20)

    if not os.getcwd().__contains__("Graphs & Photos"):
        os.chdir(os.getcwd()[:os.getcwd().index("Excel_files")] + "Graphs & Photos")
    plt.savefig(pic_name + '.png')


