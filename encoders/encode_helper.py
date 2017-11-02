import json
from helper import FileHelper
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
