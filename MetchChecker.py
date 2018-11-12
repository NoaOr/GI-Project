import numpy as np
import pandas as pd


def get_score(sentence1, sentence2):
    # change to " "
    words1 = sentence1.split(',')
    words2 = sentence2.split(',')

    set1 = set(words1)
    set2 = set(words2)
    if len(set1) > len(set2):
        max_len = len(set1)
    else:
        max_len = len(set2)

    inter = set1.intersection(set2)
    sum = 0
    for word in inter:
        index1 = words1.index(word) + 1
        index2 = words2.index(word) + 1
        sum += 1/np.power(index1, 2) + 1/np.power(index2, 2)
    precent_match = sum/(2*max_len)
    return precent_match


def compare_with_table(data_frame, col_name, sentence):
    score_list = []
    for row in range(data_frame.shape[0]):
        sentence_to_compare = data_frame.at[row, col_name]
        metch_score = get_score(sentence.upper(), sentence_to_compare.upper())
        score_list.append(metch_score)
    return score_list


def get_top_matches(df, accurecy, col_name, sentence):
    top_list = []
    score_list = compare_with_table(df, col_name, sentence)
    max_item = max(score_list)
    for i in range(len(score_list)):
        if (max_item - score_list.__getitem__(i)) < accurecy:
            top_list.append(i)

    return top_list

if __name__ == '__main__':

    sentence = 'Milk, human, mature, fluid'
    data = {'Food Description':
                ['Milk, imitation, fluid, non-soy, sweetened, flavors other than chocolate',
                 'Milk, soy, dry, reconstituted, not babys',
                 'Yogurt, vanilla, lemon, or coffee flavor, whole milk', 'milk, human']}

    df = pd.DataFrame(data, columns=['Food Description'])

    score_list = get_top_matches(df, 0.5, 'Food Description', sentence)
    print(score_list)