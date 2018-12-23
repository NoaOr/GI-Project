from Merge_GI_USDA.MetchChecker import *


def add_sentence_to_df_by_match(USDA_desc, accuracy, GI_df, GI_col_name, USDA_df, usda_row):

    cols_to_add = USDA_df.columns.values
    cols_to_add = cols_to_add[1:]

    if not 'acc' in GI_df.columns:
        GI_df['acc'] = ""
        GI_df['match-sent'] = ""

        for col_name in cols_to_add:
            GI_df[col_name] = ""

    top_dict = get_top_matches(GI_df, accuracy, GI_col_name, USDA_desc)
    for key, value in top_dict.items():
        print ("USDA_ROW: " , usda_row)
        print("USDA_desc " , USDA_desc)
        if str(GI_df.loc[key, 'acc']) < str(value) or str(GI_df.loc[key, 'acc']) == ''\
                or str(GI_df.loc[key, 'acc']) == 'nan':

            # add row 1 from usda to t1 in row 3(GI)
            for col_name in cols_to_add:
                GI_df.loc[key, col_name] = USDA_df.at[usda_row, col_name]
            GI_df.loc[key, 'acc'] = value
            GI_df.loc[key, 'match-sent'] = USDA_desc