import pandas as pd

def replace_strings_to_int():
    df = pd.read_excel("statistics.xlsx")

    string_values = df.gender.unique()
    dic = dict(zip(string_values, range(len(string_values))))
    df = df.replace(dic)

    string_values = df.glucose_tolerance_category.unique()
    dic = dict(zip(string_values, range(len(string_values))))
    df = df.replace(dic)

    write_to_excel(df)

def write_to_excel(df):
    writer = pd.ExcelWriter('statistics.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()


if __name__ == '__main__':
    replace_strings_to_int()