from MetchChecker import *


def add_sentence_to_df_by_match(sentence, accuracy, df, col_name):

    if not 'acc' in df.columns:
        df['acc'] = ""
        df['match-sent'] = ""
    top_dict = get_top_matches(df, accuracy, col_name, sentence)
    for key, value in top_dict.items():
        if str(df.loc[key, 'acc']) < str(value) or str(df.loc[key, 'acc']) == ''\
                or str(df.loc[key, 'acc']) == 'nan':
            df.loc[key, 'acc'] = value
            df.loc[key, 'match-sent'] = sentence
