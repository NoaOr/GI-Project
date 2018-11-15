from MetchChecker import *


def add_sentence_to_df_by_match(sentence, accuracy, df, col_name):

    df['acc'] = ""
    df['match-sent'] = ""
    top_dict = get_top_matches(df, accuracy, col_name, sentence)
    for key, value in top_dict.items():
        df.loc[key, 'acc'] = value
        df.loc[key, 'match-sent'] = sentence
