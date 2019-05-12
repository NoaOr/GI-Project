import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing


if __name__ == '__main__':
    os.chdir(os.getcwd()[:os.getcwd().index("Plot_Graphs")] + "Excel_files")
    gi_usda_df = pd.read_excel("test.xlsx")

    gi_usda_df = gi_usda_df.drop(['CSFII 1994-96 Food Code',
                     'source table', 'NDB_No', 'reference food & time period', 'serve Size g',
                     'available cerbo hydrate', 'GL per serve', 'GI_2', 'acc', 'match-sent',
                     'GmWt_Desc2', 'GmWt_Desc1', 'Manganese_(mg)',
                     'GmWt_1', 'GmWt_2', 'Panto_Acid_mg)', 'Choline_Tot_ (mg)', 'FdGrp_desc'], axis='columns')

    median_df = gi_usda_df.median(skipna=True, numeric_only=True)
    for column in gi_usda_df:
        if column == "Food Description in 1994-96 CSFII" or column == "FdGrp_desc":
            continue
        gi_usda_df[column] = gi_usda_df[column].fillna(median_df[column])

    print(gi_usda_df['Thiamin'].value_counts())
    print ("-------------------------------------------------")

    bins_values = np.arange(0, 10, 0.05)
    labels_values = np.arange(0, len(bins_values) - 1, 1)

    gi_usda_df['Thiamin_color']=""
    gi_usda_df['Thiamin_color'] = pd.cut(gi_usda_df.Thiamin, bins=bins_values, labels=labels_values)





    median_df = gi_usda_df.median(skipna=True, numeric_only=True)
    gi_usda_df['Thiamin_color'] = gi_usda_df['Thiamin_color'].fillna(median_df['Thiamin_color'])

    thiamin_arr = gi_usda_df['Thiamin_color']
    writer = pd.ExcelWriter('thiamin.xlsx', engine='xlsxwriter')
    gi_usda_df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

    print(gi_usda_df['Thiamin_color'].value_counts())

    c = gi_usda_df['Thiamin_color']
    plt.scatter(x='Carbohydrt_(g)', y='GI Value', c='Thiamin_color', cmap='viridis')
    plt.xlabel('Carbohydrt_(g)')
    plt.ylabel('GI Value')

    # fig = category_scatter(x='x', y='y', label_col='label',
    #                        data=df, legend_loc='upper left')

    if not os.getcwd().__contains__("Graphs & Photos"):
        os.chdir(os.getcwd()[:os.getcwd().index("Excel_files")] + "Graphs & Photos")
    plt.savefig("carbo_vs_gi_by_thiamin" + '.png')