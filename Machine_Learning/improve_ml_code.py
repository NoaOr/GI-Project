import os
import pandas as pd
from Machine_Learning import ml_code

def run_on_big_food_group(df):


    biggest_food_group = df['FdGrp_desc'].value_counts().index[0]

    ml_df = df.loc[df['FdGrp_desc'] == biggest_food_group]

    # ML_code.learn(ml_df, "biggest_fg")


def add_features_to_df(origin_df):
    new_df = origin_df.copy()
    new_df["carbo-protein"] = ""
    new_df["carbo-lipid"] = ""
    new_df["carbo-fiber_(availableCarbo)"] = ""

    for index, row in new_df.iterrows():
        carbo_val = new_df.at[index,'Carbohydrt_(g)']
        protein_val = new_df.at[index, 'Protein_(g)']
        lipid_val = new_df.at[index, 'Lipid_Tot_(g)']
        fiber_val = new_df.at[index, 'Fiber_TD_(g)']

        cp = round(carbo_val/protein_val, 3)
        cl = round(carbo_val/lipid_val, 3)
        cf = round(carbo_val/fiber_val, 3)

        new_df.loc[index, "carbo-protein"] = cp
        new_df.loc[index, "carbo-lipid"] = cl
        new_df.loc[index, "carbo-fiber_(availableCarbo)"] = cf

    ml_code.learn(new_df, pic_name="with_new_ftrs")


if __name__ == '__main__':

    if not os.getcwd().__contains__("Excel_files"):
        os.chdir(os.getcwd()[:os.getcwd().index("Machine_Learning")] + "Excel_files")
    df = pd.read_excel("GI_USDA_full.xlsx")


    # median_df = df.median(skipna=True, numeric_only=True)
    # for column in df:
    #     df[column] = df[column].fillna(median_df[column])

    # run_on_big_food_group(df)

    add_features_to_df(df)
