import os

from encoders.encode_helper import EncodeHelper
from helpers.preprocess_helper import PreprocessHelper
from os.path import basename


def alphabet_encode_and_store(source_file_name, dist_dir, alphabet, to_lower=True):
    file_name = basename(source_file_name)
    dist_full_name = dist_dir + file_name + ".pickle"

    encodings, messages, scores = EncodeHelper.encode_messages(alphabet, source_file_name, to_lower)
    PreprocessHelper.store_encoded_messages(dist_full_name, encodings, scores)


def encode_and_store(alphabet, test_folder, train_folder, to_lower):
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    alphabet_encode_and_store("data/subsampled/Beauty_test_13862.json", test_folder, alphabet, to_lower=to_lower)
    alphabet_encode_and_store("data/subsampled/Games_test_5876.json", test_folder, alphabet, to_lower=to_lower)
    alphabet_encode_and_store("data/subsampled/Grocery_test_7469.json", test_folder, alphabet, to_lower=to_lower)
    alphabet_encode_and_store("data/subsampled/Pet_Supplies_test_11340.json", test_folder, alphabet, to_lower=to_lower)
    alphabet_encode_and_store("data/subsampled/Phones_test_13895.json", test_folder, alphabet, to_lower=to_lower)

    alphabet_encode_and_store("data/subsampled/Beauty_train_32343.json", train_folder, alphabet, to_lower=to_lower)
    alphabet_encode_and_store("data/subsampled/Games_train_13709.json", train_folder, alphabet, to_lower=to_lower)
    alphabet_encode_and_store("data/subsampled/Grocery_train_17426.json", train_folder, alphabet, to_lower=to_lower)
    alphabet_encode_and_store("data/subsampled/Pet_Supplies_train_26460.json", train_folder, alphabet, to_lower=to_lower)
    alphabet_encode_and_store("data/subsampled/Phones_train_32420.json", train_folder, alphabet, to_lower=to_lower)


#-----------Standart Encoding-------------

alphabet = EncodeHelper.make_alphabet_encoding(alphabet=EncodeHelper.alphabet_standard)
test_folder = "data/encoded/standard/test/"
train_folder = "data/encoded/standard/train/"
encode_and_store(alphabet, test_folder, train_folder, True)

#-----------Standart Group Encoding-------------

alphabet = EncodeHelper.make_standart_group_encoding()
test_folder = "data/encoded/standard_group/test/"
train_folder = "data/encoded/standard_group/train/"
encode_and_store(alphabet, test_folder, train_folder, False)

#-----------ASCII Encoding-------------

alphabet = EncodeHelper.make_ascii_encoding()
print(alphabet)
test_folder = "data/encoded/ascii/test/"
train_folder = "data/encoded/ascii/train/"
encode_and_store(alphabet, test_folder, train_folder, False)

#-----------ASCII Group Encoding-------------

alphabet = EncodeHelper.make_ascii_group_encoding()
print(alphabet)
test_folder = "data/encoded/ascii_group/test/"
train_folder = "data/encoded/ascii_group/train/"
encode_and_store(alphabet, test_folder, train_folder, False)

#-----------Verify Encoding-------------

encoding_name = "ascii"
alphabet = EncodeHelper.make_ascii_encoding()

messages, _ = PreprocessHelper.get_encoded_messages("data/encoded/{}/test/Beauty_test_13862.json.pickle".format(encoding_name))
decoded = EncodeHelper.decode_messages(alphabet, messages)
print(decoded[0])
print("---------------------------------------")
print(decoded[10:30])