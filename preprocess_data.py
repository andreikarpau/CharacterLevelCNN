from encoders.encode_helper import EncodeHelper
from helper import FileHelper
import json


def filter_and_save_to_file():
    texts, summaries, scores = FileHelper.read_data_file("data/Grocery_Gourmet_Food.json", FileHelper.default_filter, 1000)

    with open('data/Grocery_Filtered_1000.json', 'w') as f:
        for i, val in enumerate(texts):
            json.dump({'reviewText': val, 'summary': summaries[i], 'overall': scores[i]}, f, ensure_ascii=False)
            f.write('\n')


encoded_data = EncodeHelper.read_encode_review_text('data/Grocery_Filtered_1000.json', EncodeHelper.get_alphabet())
print(encoded_data[0])
print(encoded_data[1])
print(encoded_data[2])
