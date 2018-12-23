import numpy as np
import pandas as pd
import difflib
from fuzzywuzzy import fuzz

counter = 0
def get_score(sentence1, sentence2):
    return fuzz.ratio(sentence1, sentence2)




    # return precent_match


def compare_with_table(data_frame, col_name, sentence):
    score_list = []
    for row in range(data_frame.shape[0]):
        sentence_to_compare = data_frame.at[row, col_name]
        metch_score = get_score(sentence.upper(), sentence_to_compare.upper())
        score_list.append(metch_score)
    return score_list


def get_top_matches(df, accurecy, col_name, sentence):
    global counter
    top_dict = {}
    score_list = compare_with_table(df, col_name, sentence)
    # print (score_list)
    max_item = max(score_list)
    for i in range(len(score_list)):
        if score_list.__getitem__(i) >= 90:
            if (max_item - score_list.__getitem__(i)) < accurecy:
                top_dict[i] = score_list.__getitem__(i)
    print (counter, top_dict)
    counter +=1
    return top_dict


# if __name__ == '__main__':
#
#     sentence = 'Milk, human, mature, fluid'
#     data = {'Food Description':
#                 ['Milk, imitation, fluid, non-soy, sweetened, flavors other than chocolate',
#                  'Milk, soy, dry, reconstituted, not babys',
#                  'Yogurt, vanilla, lemon, or coffee flavor, whole milk', 'milk, human']}
#
#     df = pd.DataFrame(data, columns=['Food Description'])
#
#     score_list = get_top_matches(df, 0.5, 'Food Description', sentence)
#     print(score_list)