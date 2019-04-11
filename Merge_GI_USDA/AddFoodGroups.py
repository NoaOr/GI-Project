import os
import pandas as pd
from fuzzywuzzy import fuzz


def merge_food_group_to_description():
    os.chdir(os.getcwd()[:os.getcwd().index("Merge_GI_USDA")] + "Excel_files/USDA_Src")
    food_groups = pd.read_excel("FD_GROUP.xlsx")
    food_groups_df = pd.DataFrame(food_groups)

    food_desc = pd.read_excel("FOOD_DES.xlsx")
    food_desc_df = pd.DataFrame(food_desc)

    food_desc_df['FdGrp_desc'] = ""

    for i in range(food_desc_df.shape[0]):
        print(i)
        for j in range(food_groups_df.shape[0]):
            if food_desc_df.iloc[i]["FdGrp_Cd"] == food_groups_df.iloc[j]["FdGrp_Cd"]:
                food_desc_df.loc[i, "FdGrp_desc"] = food_groups_df.iat[j, 1]
    writer = pd.ExcelWriter('food_groups_with_desc.xlsx', engine='xlsxwriter')
    food_desc_df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

def merge_final_gi_usda_with_food_groups():
    os.chdir(os.getcwd()[:os.getcwd().index("Merge_GI_USDA")] + "Excel_files")
    food_groups = pd.read_excel("USDA_Src/food_groups_with_desc.xlsx")
    food_groups_df = pd.DataFrame(food_groups)

    #gi_usda = pd.read_excel("GI_USDA_CLEAN_FOOD_GROUPS.xlsx")
    gi_usda = pd.read_excel("gi_usda_without_indexes.xlsx")
    gi_usda_df = pd.DataFrame(gi_usda)

    gi_usda_df['FdGrp_desc'] = ""


    # gi_usda_df.loc[229, "FdGrp_desc"] = "noa"
    #
    #
    # writer = pd.ExcelWriter('noa.xlsx', engine='xlsxwriter')
    # gi_usda_df.to_excel(writer, sheet_name='Sheet1')
    # writer.save()


    #for i in range(770, 775):
    for i in range(gi_usda_df.shape[0]):
        print(i)
        max_match = 0
        match = ""
        for j in range(food_groups_df.shape[0]):
            if gi_usda_df.iloc[i]["match-sent"] == food_groups_df.iloc[j]["Long_Desc"]:
                #gi_usda_df.loc[i, "FdGrp_desc"] = food_groups_df.iat[j, 2]
                match = food_groups_df.iat[j, 2]
                break
            else:
                match_acc = fuzz.ratio(gi_usda_df.iloc[i]["match-sent"], food_groups_df.iloc[j]["Long_Desc"])
                if match_acc > max_match:
                    match = food_groups_df.iat[j, 2]
                    max_match = match_acc

        gi_usda_df.loc[i, "FdGrp_desc"] = match
        print("__________________________________")
        print("gi-usda: ", gi_usda_df.iat[i, 10])
        print("desc: ", match)
        print("__________________________________")


    writer = pd.ExcelWriter('GI_USDA_CLEAN_FOOD_GROUPS.xlsx', engine='xlsxwriter')
    gi_usda_df.to_excel(writer, sheet_name='Sheet1')
    writer.save()


if __name__ == '__main__':
    #merge_food_group_to_description()
    merge_final_gi_usda_with_food_groups()