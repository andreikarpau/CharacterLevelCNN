from encoders.encode_helper import EncodeHelper
from preprocess.helper import FileHelper
import json
import pickle


class PreprocessData:
    @staticmethod
    def save_filtered_messages():
        texts, summaries, scores = FileHelper.read_data_file("data/Grocery_Gourmet_Food.json",
                                                             FileHelper.default_filter, 1000)

        with open('data/Grocery_Filtered_1000.json', 'w') as f:
            for i, val in enumerate(texts):
                json.dump({'reviewText': val, 'summary': summaries[i], 'overall': scores[i]}, f, ensure_ascii=False)
                f.write('\n')

    @staticmethod
    def store_encoded_messages(filename='data/Grocery_Filtered_1000_Encoded.pickle'):
        PreprocessData.save_filtered_messages()
        encoded_messages, messages, scores = EncodeHelper.encode_messages(EncodeHelper.make_alphabet_encoding())
        items = [None] * len(encoded_messages)

        with open(filename, 'wb') as f:
            for i, val in enumerate(encoded_messages):
                items[i] = {'encodedText': val, 'overall': scores[i]}

            pickle.dump(items, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def get_encoded_messages(filename='data/Grocery_Filtered_1000_Encoded.pickle'):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            messages = [None] * len(data)
            scores = [None] * len(data)
            i = 0

            for d in data:
                messages[i] = d['encodedText']
                scores[i] = d['overall']
                i += 1

            return messages, scores


#PreprocessData.store_encoded_messages()
messages, scores = PreprocessData.get_encoded_messages()
