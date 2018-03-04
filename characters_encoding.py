from encoders.encode_helper import EncodeHelper
from preprocess.preprocess_helper import PreprocessHelper
from os.path import basename


def alphabet_encode_and_store(source_file_name, dist_dir, alphabet):
    file_name = basename(source_file_name)
    dist_full_name = dist_dir + file_name + ".pickle"

    encodings, messages, scores = EncodeHelper.encode_messages(alphabet, source_file_name, True)
    PreprocessHelper.store_encoded_messages(dist_full_name, encodings, scores)


alphabet = EncodeHelper.make_alphabet_encoding(alphabet=EncodeHelper.alphabet_standard)
alphabet_encode_and_store("data/subsampled/Beauty_test_13862.json", "data/encoded/standart/test/", alphabet)
alphabet_encode_and_store("data/subsampled/Games_test_5876.json", "data/encoded/standart/test/", alphabet)
alphabet_encode_and_store("data/subsampled/Grocery_test_7469.json", "data/encoded/standart/test/", alphabet)
alphabet_encode_and_store("data/subsampled/Pet_Supplies_test_11340.json", "data/encoded/standart/test/", alphabet)
alphabet_encode_and_store("data/subsampled/Phones_test_13895.json", "data/encoded/standart/test/", alphabet)

alphabet_encode_and_store("data/subsampled/Beauty_train_32343.json", "data/encoded/standart/train/", alphabet)
alphabet_encode_and_store("data/subsampled/Games_train_13709.json", "data/encoded/standart/train/", alphabet)
alphabet_encode_and_store("data/subsampled/Grocery_train_17426.json", "data/encoded/standart/train/", alphabet)
alphabet_encode_and_store("data/subsampled/Pet_Supplies_train_26460.json", "data/encoded/standart/train/", alphabet)
alphabet_encode_and_store("data/subsampled/Phones_train_32420.json", "data/encoded/standart/train/", alphabet)


# messages, _ = PreprocessHelper.get_encoded_messages("data/encoded/standart/test/Beauty_test_13862.json.pickle")
# decoded = EncodeHelper.decode_messages(alphabet, messages)
# print(decoded)