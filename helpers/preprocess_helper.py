import os
import numpy as np

from helpers.file_helper import FileHelper
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
    def get_encoded_messages_from_folder(folder_name):
        files = os.listdir(folder_name)

        all_messages = []
        all_scores = []

        for f in files:
            messages, scores = PreprocessHelper.get_encoded_messages(folder_name + "/" + f)
            all_messages.extend(messages)
            all_scores.extend(scores)

        return all_messages, all_scores

    @staticmethod
    def bootstrap_test_indexes(reviews_count=52442, sample_size=5000, samples_number=20):
        all_indexes = []
        for sn in range(samples_number):
            indexes = None
            for i in range(sample_size):
                indexes = np.random.choice(reviews_count, sample_size)

            all_indexes.append(indexes)

        return all_indexes

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

