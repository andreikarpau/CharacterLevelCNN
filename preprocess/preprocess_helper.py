from preprocess.file_helper import FileHelper
import json
import pickle


class PreprocessHelper:
    @staticmethod
    def save_filtered_messages():
        texts, summaries, scores = FileHelper.read_data_file("data/Grocery_Gourmet_Food.json",
                                                             FileHelper.default_filter, 1000)

        with open('data/Grocery_Filtered_1000.json', 'w') as f:
            for i, val in enumerate(texts):
                json.dump({'reviewText': val, 'summary': summaries[i], 'overall': scores[i]}, f, ensure_ascii=False)
                f.write('\n')

    @staticmethod
    def store_encoded_messages(file_name, encodings, scores):
        items = [None] * len(encodings)

        with open(file_name, 'wb') as f:
            for i, encoding in enumerate(encodings):
                items[i] = {'encodedText': encoding, 'overall': scores[i]}

            pickle.dump(items, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def get_encoded_messages(file_name):
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
            messages = [None] * len(data)
            scores = [None] * len(data)
            i = 0

            for d in data:
                messages[i] = d['encodedText']
                scores[i] = d['overall']
                i += 1

            return messages, scores

