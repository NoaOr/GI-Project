import pandas as pd
import os

def correlation_matrix(df):
    col_1 = "Water_(g)"
    col_2 = "Carbohydrt_(g)"

    df_1 = pd.DataFrame(df.loc[:, col_1])
    df_2 = pd.DataFrame(df.loc[:, col_2])
    corr = df_1.corrwith(df_2)
    print (corr)

if __name__ == '__main__':
    os.chdir(os.getcwd()[:os.getcwd().index("Correlation")] + "Excel_files")
    USDA = pd.read_excel("USDA_data.xlsx")
    usda_df = pd.DataFrame(USDA)
    correlation_matrix(usda_df)