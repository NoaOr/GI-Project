import numpy
import pandas as pd

def compare_with_table(data_frame,col_name, sentence):
    for row in data_frame:
        sentence_to_compare = data_frame.loc[[row], [col_name]]
