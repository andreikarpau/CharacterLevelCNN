import json

import numpy as np


class FileHelper:
    @staticmethod
    def default_filter(text, summary, score):
        # if score < 1 or score == 3 or 5 < score:
        #     return False

        if len(text) < 100 or 1024 < (len(text) + len(summary) + len(" \n ")):
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
    def encode_to_ascii(text):
        for code in map(ord, text):
            print(bin(int(code))[2:].zfill(8))

    @staticmethod
    def encode_to_alphabet(text, alphabet_dict, to_lower=False):
        length = len(text)
        encoded_list = [None] * length

        if to_lower:
            text = text.lower()

        i = 0
        for c in text:
            if c in alphabet_dict:
                encoded_list[i] = alphabet_dict[c]
            else:
                encoded_list[i] = alphabet_dict['blank']

            i += 1

        return encoded_list

    # @staticmethod
    # def encode_from_alphabet(encoded_message, alphabet_dict, to_lower=False):
    #     for c in encoded_message:
    #         if c in alphabet_dict:
    #             encoded_list[i] = alphabet_dict[c]
    #         else:
    #             encoded_list[i] = alphabet_dict['blank']
    #
    #         i += 1
    #
    #     return encoded_list

    alphabet_standard = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’/\|_@#$%ˆ&*˜‘+-=<>()[]{}\n"

    @staticmethod
    def make_alphabet_encoding(alphabet=alphabet_standard):
        codes = {}
        length = len(alphabet)
        index = 0

        for c in alphabet:
            code_array = [0] * length
            code_array[index] = 1
            index += 1
            codes[c] = code_array

        codes[' '] = [0] * length
        codes['blank'] = [0] * length
        return codes

    @staticmethod
    def read_word2vec(file_name="word2vec/glove.6B.50d.txt"):
        with open(file_name, "rb") as lines:
            w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
                   for line in lines}
            return w2v
