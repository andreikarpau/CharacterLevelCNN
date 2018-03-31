import json
from helpers.file_helper import FileHelper
import string


class EncodeHelper:
    alphabet_standard = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’/\|_@#$%ˆ&*˜‘+-=<>()[]{}\n"

    @staticmethod
    def make_standart_group_encoding():
        standart_length = len(EncodeHelper.alphabet_standard)
        standart_encoding = EncodeHelper.make_alphabet_encoding(EncodeHelper.alphabet_standard, standart_length + 4)

        letters = list(string.ascii_lowercase)
        for l in letters:
            code = standart_encoding[l]
            code[standart_length] = 1.0
            upper_code = list(code)
            upper_code[standart_length + 1] = 1.0
            standart_encoding[l.upper()] = upper_code

        for i in range(0,10):
            digit = str(i)
            code = standart_encoding[digit]
            code[standart_length + 2] = 1.0

        for spec in list("-,;.!?:’/\|_@#$%ˆ&*˜‘+-=<>()[]{}"):
            code = standart_encoding[spec]
            code[standart_length + 3] = 1.0

        return standart_encoding

    @staticmethod
    def make_alphabet_encoding(alphabet, encoding_length=None):
        codes = {}

        length = encoding_length
        if encoding_length is None:
            length = len(alphabet)

        index = 0

        for c in alphabet:
            code_array = [0.0] * length
            code_array[index] = 1.0
            index += 1
            codes[c] = code_array

        codes['blank'] = [0] * length
        codes[' '] = [0] * length

        return codes

    @staticmethod
    def make_back_alphabet(alphabet):
        back_alphabet = {}

        for char, encoding in alphabet.items():
            back_alphabet[str(encoding)] = char

        return back_alphabet

    @staticmethod
    def encode_messages(alphabet_dict, file_name, to_lower, size=1024, max_num=None):
        texts, scores = FileHelper.read_message_scores_from_file(file_name, max_num)

        messages = [None] * len(texts)
        encoded_messages = [None] * len(texts)

        for i in range(0, len(texts)):
            message = texts[i]

            if size is not None:
                spaces = " " * (size - len(message))
                message = message + spaces

            messages[i] = message
            encoded_messages[i] = EncodeHelper.encode_to_alphabet(message, alphabet_dict, to_lower)

        return encoded_messages, messages, scores

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
    def decode_messages(alphabet, messages):
        back_alphabet = EncodeHelper.make_back_alphabet(alphabet)
        decoded_messages = []

        for message in messages:
            decoded_message = ""

            for encoded_char in message:
                char = back_alphabet[str(encoded_char)]
                decoded_message += char

            decoded_messages.append(decoded_message)

        return decoded_messages