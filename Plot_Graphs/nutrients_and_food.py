import plotly.plotly as py
import cufflinks as cf
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    os.chdir(os.getcwd()[:os.getcwd().index("Plot_Graphs")] + "Excel_files")
    df = pd.read_excel("GI_USDA_clean.xlsx")

    ml_df = df.drop(['CSFII 1994-96 Food Code', 'Food Description in 1994-96 CSFII',
                     'source table', 'reference food & time period', 'serve Size g',
                     'available cerbo hydrate', 'GL per serve', 'GI_2', 'acc', 'match-sent',
                     'GmWt_Desc2', 'GmWt_Desc1', 'Manganese_(mg)',
                     'GmWt_1', 'GmWt_2', 'Panto_Acid_mg)', 'Choline_Tot_ (mg)'], axis='columns')

    median_df = ml_df.median(skipna=True, numeric_only=True)
    for column in ml_df:
        ml_df[column] = ml_df[column].fillna(median_df[column])

    ax = sns.heatmap(ml_df, linewidths=0.00001)

    os.chdir(os.getcwd()[:os.getcwd().index("Excel_files")] + "Graphs & Photos")

    figure = ax.get_figure()
    figure.savefig('output.png', dpi=400)