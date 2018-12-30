from Merge_GI_USDA.MetchChecker import *


def add_sentence_to_df_by_match(GI_desc, accuracy, usda_df, USDA_col_name, GI_df, GI_row):

    cols_to_add = usda_df.columns.values
    cols_to_add = cols_to_add[1:]


    if not 'acc' in usda_df.columns:
        GI_df['acc'] = ""
        GI_df['match-sent'] = ""

        for col_name in cols_to_add:
            GI_df[col_name] = ""

    top_dict = get_top_matches(usda_df, accuracy, USDA_col_name, GI_desc)
    for key, value in top_dict.items():

        if str(GI_df.loc[GI_row, 'acc']) < str(value) or str(GI_df.loc[GI_row, 'acc']) == ''\
                or str(GI_df.loc[GI_row, 'acc']) == 'nan':

            print("---------------------------------------------------------------------------------")
            print("USDA_ROW: ", GI_row)
            print("USDA_desc ", usda_df.at[key, USDA_col_name])
            print("GI_desc ", GI_desc)
            print("---------------------------------------------------------------------------------")


            # add row 1 from usda to t1 in row 3(GI)
            for col_name in cols_to_add:
                GI_df.loc[GI_row, col_name] = usda_df.at[key, col_name]
            GI_df.loc[GI_row, 'acc'] = value
            GI_df.loc[GI_row, 'match-sent'] = usda_df.at[key, USDA_col_name]