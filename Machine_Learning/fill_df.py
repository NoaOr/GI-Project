import os
import pandas as pd
from prettytable import PrettyTable


if __name__ == '__main__':
    os.chdir(os.getcwd()[:os.getcwd().index("Machine_Learning")] + "Excel_files")
    df = pd.read_excel("GI_USDA_CLEAN_FOOD_GROUPS.xlsx")

    null_columns = df.columns[df.isnull().any()]
    print(df[df["Sugar_Tot_(g)"].isna()]['FdGrp_desc'])

    ml_df = df.drop(['CSFII 1994-96 Food Code',
                     'source table', 'NDB_No', 'reference food & time period', 'serve Size g',
                     'available cerbo hydrate', 'GL per serve', 'GI_2', 'acc', 'match-sent',
                     'GmWt_Desc2', 'GmWt_Desc1', 'Manganese_(mg)',
                     'GmWt_1', 'GmWt_2', 'Panto_Acid_mg)', 'Choline_Tot_ (mg)'], axis='columns')

    food_groups_df = pd.read_excel("GI_USDA_CLEAN_FOOD_GROUPS.xlsx")
    food_groups = food_groups_df.pop("FdGrp_desc")
    food_groups = food_groups.unique()

    median_table = PrettyTable()
    cols = list(ml_df.columns.values)
    cols.remove("Food Description in 1994-96 CSFII")
    cols.remove("FdGrp_desc")
    cols.insert(0, "FdGrp_desc")

    median_table.field_names = cols
    for food_group in food_groups:
        row = [food_group]
        for column in ml_df:
            if column == "Food Description in 1994-96 CSFII" or column == "FdGrp_desc":
                continue
            m1 = (ml_df['FdGrp_desc'] == food_group)
            median = df.loc[m1, column].median()
            row.append(median)
            ml_df.loc[m1, column] = df.loc[m1, column].fillna(df.loc[m1, column].median())
        median_table.add_row(row)

    data = median_table.get_string()

    with open('median_values.txt', 'w') as f:
        f.write(data)

    writer = pd.ExcelWriter('GI_USDA_full.xlsx', engine='xlsxwriter')
    ml_df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

