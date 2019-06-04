import pandas as pd


def delete_duplicate_peaks(df):
    """
    Delete duplicate peaks in data set
    :param df:
    :return:
    """
    df = df.drop_duplicates(subset='peak_start_instant', keep=False)
    write_to_excel(df)


def delete_activity_rows():
    """
    The function deletes the activiaty rows
    :return:
    """
    df = pd.read_excel("dataset.xlsx")

    df = df[df.row_type != 'PEAK_AND_ACTIVITY']
    write_to_excel(df)


def replace_strings_to_int():
    """
    The function replaces categorial values in numerical values
    :return:
    """
    df = pd.read_excel("dataset.xlsx")

    string_values = df.day_part.unique()
    dic = dict(zip(string_values, range(len(string_values))))
    df = df.replace(dic)

    string_values = df.meal_type.unique()
    dic = dict(zip(string_values, range(len(string_values))))
    df = df.replace(dic)

    write_to_excel(df)


def write_to_excel(df):
    """
    Writes the received data frame to excel file
    :param df:
    :return:
    """
    writer = pd.ExcelWriter('dataset.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()


if __name__ == '__main__':
    df = pd.read_excel("origin_dataset.xlsx")
    # delete_duplicate_peaks(df)
    # delete_activity_rows()
    replace_strings_to_int()

    #PEAK_AND_ACTIVITY
    #food_names
