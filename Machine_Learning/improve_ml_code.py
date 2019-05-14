import os
import pandas as pd

def run_on_big_food_group():
    os.chdir(os.getcwd()[:os.getcwd().index("Machine_Learning")] + "Excel_files")
    df = pd.read_excel("GI_USDA_full.xlsx")

    biggest_food_group = df['FdGrp_desc'].value_counts().index[0]

    ml_df = df.loc[df['FdGrp_desc'] == biggest_food_group]

    # ML_code.learn(ml_df, "biggest_fg")




if __name__ == '__main__':
    run_on_big_food_group()