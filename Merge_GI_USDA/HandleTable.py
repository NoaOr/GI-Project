from Merge_GI_USDA.MetchChecker import *
import Merge_GI_USDA.Main as merge_main
import config as cfg

num_matches = 0

def add_sentence_to_df_by_match(GI_desc, accuracy, usda_df, USDA_col_name, GI_row):
    global num_matches
    cols_to_add = usda_df.columns.values
    cols_to_add = cols_to_add[1:]


    if not 'acc' in cfg.GI_df_2:
        cfg.GI_df_2['acc'] = ""
        cfg.GI_df_2['match-sent'] = ""

        for col_name in cols_to_add:
            cfg.GI_df_2[col_name] = ""

    top_dict = get_top_matches(usda_df, accuracy, USDA_col_name, GI_desc)
    for key, value in top_dict.items():

        if str(cfg.GI_df_2.loc[GI_row, 'acc']) < str(value) or str(cfg.GI_df_2.loc[GI_row, 'acc']) == ''\
                or str(cfg.GI_df_2.loc[GI_row, 'acc']) == 'nan':
            num_matches += 1

            print("---------------------------------------------------------------------------------")
            print("GI_ROW: ", GI_row)
            print("USDA_desc ", usda_df.at[key, USDA_col_name])
            print("GI_desc ", GI_desc)
            print("num matches = ", num_matches)
            print("---------------------------------------------------------------------------------")


            # add row 1 from usda to t1 in row 3(GI)
            for col_name in cols_to_add:
                cfg.GI_df_2.loc[GI_row, col_name] = usda_df.at[key, col_name]
            cfg.GI_df_2.loc[GI_row, 'acc'] = value
            cfg.GI_df_2.loc[GI_row, 'match-sent'] = usda_df.at[key, USDA_col_name]

