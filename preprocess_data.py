from helper import FileHelper
import json

texts, summaries, scores = FileHelper.read_data_file("data/Grocery_Gourmet_Food.json", FileHelper.default_filter, 1000)

with open('data/Grocery_Filtered_1000.json', 'w') as f:
    for i, val in enumerate(texts):
        json.dump({'reviewText': val, 'summary': summaries[i], 'overall': scores[i]}, f, ensure_ascii=False)
        f.write('\n')


