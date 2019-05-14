from fuzzywuzzy import fuzz


if __name__ == '__main__':

    arr = ["Milk, filled, fluid, with blend of hydrogenated vegetable oils",
"Milk, filled, fluid, with lauric acid oil",
"Milk, fluid, 1% fat, without added vitamin A and vitamin D",
"Milk, fluid, nonfat, calcium fortified (fat free or skim)",
"Milk, goat, fluid, with added vitamin D",
"Milk, human, mature, fluid",
"Milk, imitation, non-soy",
"Milk, indian buffalo, fluid",
"Milk, low sodium, fluid",
"Milk, lowfat, fluid, 1% milkfat, protein fortified, with added vitamin A and vitamin D",
"Milk, lowfat, fluid, 1% milkfat, with added nonfat milk solids, vitamin A and vitamin D",
"Milk, lowfat, fluid, 1% milkfat, with added vitamin A and vitamin D",
"Milk, nonfat, fluid, protein fortified, with added vitamin A and vitamin D (fat free and skim)",
"Milk, nonfat, fluid, with added nonfat milk solids, vitamin A and vitamin D (fat free or skim)",
"Milk, nonfat, fluid, with added vitamin A and vitamin D (fat free or skim)",
"Milk, nonfat, fluid, without added vitamin A and vitamin D (fat free or skim)",
"Milk, producer, fluid, 3.7% milkfat",
"Milk, reduced fat, fluid, 2% milkfat, protein fortified, with added vitamin A and vitamin D",
"Milk, reduced fat, fluid, 2% milkfat, with added nonfat milk solids and vitamin A and vitamin D",
"Milk, reduced fat, fluid, 2% milkfat, with added vitamin A and vitamin D",
"Milk, reduced fat, fluid, 2% milkfat, with added nonfat milk solids, without added vitamin A",
"Milk, reduced fat, fluid, 2% milkfat, without added vitamin A and vitamin D",
"Milk, sheep, fluid",
"Milk, whole, 3.25% milkfat, with added vitamin D",
"Milk, whole, 3.25% milkfat, without added vitamin A and vitamin D"]


    # sentence1 = "Milk, cow's, fluid, lactose reduced, 1% fat, fortified with"
    # sentence1 = "Milk, cow's, fluid, whole"
    sentence1 = "Milk, cow's, fluid, 2% fat"
    for val in arr:
        sentence2 = val
        print("------------------------------")
        print(val)
        print(fuzz.ratio(sentence1, sentence2))
        print("------------------------------")

