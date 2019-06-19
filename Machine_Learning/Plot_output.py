import matplotlib.pyplot as plt
import pydotplus
import sklearn
from sklearn.metrics import mean_absolute_error
import os
import matplotlib.patheffects as path_effects
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import seaborn as sns
import numpy as np
import math

font = {'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 40,
        }


def plot_graph(X_test, y_test, predict, pic_name, dir, coefficients_str=""):
    if not os.getcwd().__contains__("Graphs & Photos"):
        os.chdir(os.getcwd()[:os.getcwd().index("GI-Project")] + "GI-Project/Graphs & Photos")

    if dir != "":
        path_dir = "./" + dir

        if not os.getcwd().__contains__(dir):
            if (not os.path.isdir(path_dir)):
                os.mkdir(dir)
            os.chdir(dir)

    plt.clf()
    plt.figure(figsize=(20, 13))

    # plt.title(pic_name, fontdict=font)


    plt.scatter(y_test, predict, color='blue', s=40)
    x = list(range(10, 100))
    plt.plot(x, x, 'r--', linewidth=2)

    # plt.scatter(X_test['Carbohydrt_(g)'], y_test, color='blue', s = 75)
    # plt.scatter(X_test['Carbohydrt_(g)'], predict, color='red', s = 75)

    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)

    # plt.legend(('GI vlaue', 'predict GI value'),
    #            shadow=True, loc=(0.7, 0.85), handlelength=1.5, fontsize=25)

    if coefficients_str == "":
        # plt.xlabel('Carbohydrt' + '\n\n' +
        plt.xlabel('Measured GI' + '\n\n' +
                   'MAE = ' + str("%.3f" % mean_absolute_error(y_test, predict)) +
                   ' RMSE = ' + str("%.3f" % math.sqrt(sklearn.metrics.mean_squared_error(y_test, predict))) +
                    ' R2 = ' + str("%.3f" % sklearn.metrics.r2_score(y_test, predict)),
                    fontsize = 20)
    else:
        # plt.xlabel('Carbohydrt' + '\n\n' +
        plt.xlabel('Measured GI' + '\n\n' +
                   'MAE = ' + str("%.3f" % mean_absolute_error(y_test, predict)) +
                   ' RMSE = ' + str("%.3f" % math.sqrt(sklearn.metrics.mean_squared_error(y_test, predict))) +
                   ' R2 = ' + str("%.3f" % sklearn.metrics.r2_score(y_test, predict)) +
                   " coefficients: " + coefficients_str,
                    fontsize = 20)

    # plt.ylabel('GI value', fontsize = 20)
    plt.ylabel('Predicted GI', fontsize=20)

    if not os.getcwd().__contains__("Graphs & Photos"):
        os.chdir(os.getcwd()[:os.getcwd().index("Excel_files")] + "Graphs & Photos")
    plt.savefig(pic_name + '.png')

def plot_coefficients(coefficients_str, pic_name, dir):
    if not os.getcwd().__contains__("Graphs & Photos"):
        os.chdir(os.getcwd()[:os.getcwd().index("GI-Project")] + "GI-Project/Graphs & Photos")

    if dir != "":
        path_dir = "./" + dir

        if not os.getcwd().__contains__(dir):
            if (not os.path.isdir(path_dir)):
                os.mkdir(dir)
            os.chdir(dir)

    plt.clf()

    fig = plt.figure(figsize=(5, 12))

    text = fig.text(0.5, 0.5, coefficients_str,
                    ha='center', va='center', size=9)
    text.set_path_effects([path_effects.Normal()])

    if not os.getcwd().__contains__("Graphs & Photos"):
        os.chdir(os.getcwd()[:os.getcwd().index("Excel_files")] + "Graphs & Photos")
    pic_name = "coefficients_" + pic_name
    plt.savefig(pic_name + '.png')

def plot_two_cols(x, y, df, pic_name, dir):
    if not os.getcwd().__contains__("Graphs & Photos"):
        os.chdir(os.getcwd()[:os.getcwd().index("GI-Project")] + "GI-"
                                                                 "Project/Graphs & Photos")

    if dir != "":
        path_dir = "./" + dir

        if not os.getcwd().__contains__(dir):
            if (not os.path.isdir(path_dir)):
                os.mkdir(dir)
            os.chdir(dir)

    plt.clf()
    plt.figure(figsize=(20, 13))

    plt.title(pic_name, fontdict=font)

    plt.scatter(df[x], df[y], color='blue', s=40)

    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)

    plt.xlabel(x, fontsize=20)
    plt.ylabel(y, fontsize=20)

    plt.savefig(pic_name + '.png')


def plot_DT_nodes(DTL_model, labels, dir):
    if not os.getcwd().__contains__("Graphs & Photos"):
        os.chdir(os.getcwd()[:os.getcwd().index("GI-Project")] + "GI-Project/Graphs & Photos")

    if dir != "":
        path_dir = "./" + dir

        if not os.getcwd().__contains__(dir):
            if (not os.path.isdir(path_dir)):
                os.mkdir(dir)
            os.chdir(dir)

    dot_data = StringIO()
    export_graphviz(DTL_model, out_file=dot_data, feature_names=labels,
                    filled=True, rounded=True,
                    special_characters=False)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

    graph.set_size('"45,20!"')

    graph.write_png('DT nodes new test.png')


def plot_variable_importance(best_random, lables, pic_name1, dir):
    if not os.getcwd().__contains__("Graphs & Photos"):
        os.chdir(os.getcwd()[:os.getcwd().index("GI-Project")] + "GI-Project/Graphs & Photos")

    if dir != "":
        path_dir = "./" + dir

        if not os.getcwd().__contains__(dir):
            if (not os.path.isdir(path_dir)):
                os.mkdir(dir)
            os.chdir(dir)

    plt.clf()
    feature_imp = best_random.feature_importances_

    features_dict = dict(zip(lables, feature_imp))
    print(features_dict)

    new_features_dict = {key: val for key, val in features_dict.items() if val >= features_dict['Vit_K_(µg)']}

    indices = np.argsort(feature_imp)

    plt.title('Random Forest - Feature Importance')
    # font = {'family': 'serif',
    #         'color': 'black',
    #         'weight': 'normal',
    #         'size': 16,
    #         }

    sns.set(font_scale=0.2)


    sns.set_context("paper", rc={"font.size": 2, "axes.titlesize": 2, "axes.labelsize": 2})

    labels_list = []
    for item in new_features_dict.keys():
        labels_list.append(str(item))

    importance_list = []
    for imp in new_features_dict.values():
        importance_list.append(imp)

    # labels_list.remove('iron')
    # sns.barplot(x=feature_imp, y=labels_list)
    sns.barplot(x=importance_list, y=labels_list)

    # Add labels to your graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    # plt.title("Visualizing Important Features")
    plt.legend()
    plt.gcf().set_size_inches(15, 9.3, forward=True)
    plt.savefig(pic_name1 + '.png')

    plt.close()


