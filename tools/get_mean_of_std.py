import pandas as pd
import os


if __name__ == '__main__':
    if not os.getcwd().__contains__("Excel_files/GI_tables"):
        os.chdir(os.getcwd()[:os.getcwd().index("tools")] + "Excel_files/GI_tables")
    df = pd.read_excel("GI_Src_2_temp.xlsx")

    # df = df['GI (Glucose = 100)']
    # for index, row in df.iterrows():
    #     std = row[0].split("±")[1]
    # x = df.mean(skipna=True, axis=0)
    # print(x)

    df['std'] = df['item'].str.split(" ").str[1]

    # df['std'] = df['GI (Glucose = 100)'].str.split('1').str[1]

    std_df = pd.DataFrame(df['std'])
    x = std_df.mean(skipna=True, axis=0)
    print(x)


    print(std_df)
    mean = pd.DataFrame(std_df['std']).mean(skipna=True)
    print(mean)
    print(df['std'])
    print("Tt")
    # df['std'] = df['location'].str.split(',').str[0]‏

