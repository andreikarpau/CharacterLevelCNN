import json
from preprocess.helper import FileHelper
import string


class EncodeHelper:
    @staticmethod
    def read_encode_review_text(name, alphabet):
        texts, summaries, scores = FileHelper.read_data_file(name, None)

        results = []
        for i, val in enumerate(texts):
            encoded_text = FileHelper.encode_to_alphabet(val, alphabet)
            new_obj = {'reviewText': encoded_text, 'overall': scores[i]}
            results.append(new_obj)

        return results

    @staticmethod
    def get_alphabet(name='alphabets/ascii_printable.json'):
        with open(name, "r") as json_file:
            alphabet = json.load(json_file)
            return alphabet

    @staticmethod
    def save_ascii_alphabet_to_file(name='alphabets/ascii_printable.json'):
        printable_chars = string.printable
        codes = dict()

        for c in printable_chars:
            codes[c] = list(map(int, (list(bin(int(ord(c)))[2:].zfill(8)))))

        with open(name, 'w') as f:
            json.dump(codes, f)

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
            code_array = [0.0] * length
            code_array[index] = 1.0
            index += 1
            codes[c] = code_array

        codes[' '] = [0] * length
        codes['blank'] = [0] * length
        return codes


    @staticmethod
    def encode_messages(alphabet_dict, filename="data/Grocery_Filtered_1000.json", size=1024):
        texts, summaries, scores = FileHelper.read_data_file(filename, FileHelper.default_filter)

        messages = [None] * len(texts)
        encoded_messages = [None] * len(texts)

        for i in range(0, len(texts)):
            message = "{0}\n{1}".format(summaries[i].upper(), texts[i])
            if size is not None:
                if size < len(message):
                    message = message[:size]
                else:
                    spaces = " " * (size - len(message))
                    message = message + spaces

            messages[i] = message
            encoded_messages[i] = EncodeHelper.encode_to_alphabet(message, alphabet_dict, True)

        return encoded_messages, messages, scores

