import json

import numpy as np


class FileHelper:
    @staticmethod
    def default_filter(text, summary, score):
        # if score < 1 or score == 3 or 5 < score:
        #     return False

        if len(text) < 100 or 1024 < (len(text) + len(summary) + len("text: ; summary: ;")):
            return False

        return True

    @staticmethod
    def read_data_file(name, filter_func=None, max_num=None):
        texts = []
        summaries = []
        overall = []

        iteration = 0
        with open(name, "r") as json_file:
            for line in json_file:
                iteration = iteration + 1
                if max_num is not None and max_num <= iteration:
                    break

                str = json.loads(line)
                review_text = str["reviewText"]
                summary = str["summary"]
                score = str["overall"]

                if filter_func is not None and not filter_func(review_text, summary, score):
                    continue

                texts.append(review_text)
                summaries.append(summary)
                overall.append(score)

        return texts, summaries, overall

    @staticmethod
    def read_json_data_obj(name):
        results = []

        with open(name, "r") as json_file:
            for line in json_file:
                str = json.loads(line)
                results.append({str["reviewText"], str["summary"], str["overall"]})

        return results

    @staticmethod
    def encode_to_alphabet(text, alphabet):
        encoded_text = []

        for char in text:
            if char not in alphabet:
                print("unknown char {0}".format(char))

            encoded_text.append(alphabet[char])

        return encoded_text

    @staticmethod
    def read_word2vec(file_name="word2vec/glove.6B.50d.txt"):
        with open(file_name, "rb") as lines:
            w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
                   for line in lines}
            return w2v
