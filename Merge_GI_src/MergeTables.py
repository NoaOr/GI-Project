import pandas as pd
import Excel_files as xl
from fuzzywuzzy import fuzz
import math
import os

def is_contains_sentence(sentence, df, col_name):
    num = df.shape[0]
    for row in range(df.shape[0]):
        #print ('row: ', row)
        sentence_to_compare = df.at[row, col_name]
        if not isinstance(sentence, str) or not isinstance(sentence_to_compare, str):
            return  False, 0
        if fuzz.ratio(sentence, sentence_to_compare) == 100:
            return True, row
    return False, 0


def add_to_table(df_src, df_dst, row):
       df_dst.loc[df_dst.shape[0] + 1] = ['-', df_src.at[row, 'item'], df_src.at[row, 'GI (Glucose = 100)'],
                           '2', df_src.at[row, 'reference food & time period'],
                           df_src.at[row, 'serve Size g'], df_src.at[row, 'available cerbo hydrate'],
                           df_src.at[row, 'GL per serve'], '-']


def merge_row_in_table(t1_row, t2_row, merge_df, t2_df):
    merge_df.at[t1_row, 'source table'] = '1,2'
    merge_df.at[t1_row, 'reference food & time period'] = t2_df.loc[t2_row]['reference food & time period']
    merge_df.at[t1_row, 'serve Size g'] = t2_df.loc[t2_row]['serve Size g']
    merge_df.at[t1_row, 'available cerbo hydrate'] = t2_df.loc[t2_row]['available cerbo hydrate']
    merge_df.at[t1_row, 'GL per serve'] = t2_df.loc[t2_row]['GL per serve']
    merge_df.at[t1_row, 'GI_2'] = t2_df.loc[t2_row]['GI (Glucose = 100)']


if __name__ == '__main__':
    t1 = pd.read_excel('Excel_files/GI_tables/GI_Src_1.xlsx')
    t2 = pd.read_excel('Excel_files/GI_tables/GI_Src_2_new.xlsx')
    t1_df = pd.DataFrame(t1)
    t2_df = pd.DataFrame(t2)
    merge_df = pd.DataFrame(t1)
    merge_df['source table'] = '1'

    merge_df['reference food & time period'] = "-"
    merge_df['serve Size g'] = "-"
    merge_df['available cerbo hydrate'] = "-"
    merge_df['GL per serve'] = "-"
    merge_df['GI_2'] = "-"

    merge_df.columns = ['CSFII 1994-96 Food Code', 'Food Description in 1994-96 CSFII','GI Value', 'source table', 'reference food & time period',  'serve Size g', 'available cerbo hydrate', 'GL per serve','GI_2']

    num_rows_t1 = t1_df.shape[0]
    num_rows_t2 = t2_df.shape[0]
    col_name_t2 = 'item'
    t1_col_name = 'Food Description in 1994-96 CSFII'

    for t2_row in range(num_rows_t2 - 1):
    #for t2_row in range(549,550):
        print (t2_row)
        sentence = t2_df.loc[t2_row, col_name_t2]
        is_appear, t1_row = is_contains_sentence(sentence, t1_df, t1_col_name)
        if not is_appear:
            add_to_table(t2_df, merge_df, t2_row)
        else:
           merge_row_in_table(t1_row, t2_row, merge_df, t2_df)

    writer = pd.ExcelWriter('Excel_files/GI_tables/GI_merge.xlsx', engine='xlsxwriter')
    # Convert the dataframe to an XlsxWriter Excel object.
    merge_df.to_excel(writer, sheet_name='Sheet1')
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()