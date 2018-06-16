from sklearn.model_selection import train_test_split

from helpers.file_helper import FileHelper
import matplotlib.pyplot as plt
import numpy as np


def get_undersampled_majorities(texts, scores):
    s_index = [None] * 5
    scores = np.asarray(scores)
    texts = np.asarray(texts)
    s_index[0] = np.where(scores == 1)
    s_index[1] = np.where(scores == 2)
    s_index[2] = np.where(scores == 3)
    s_index[3] = np.where(scores == 4)
    s_index[4] = np.where(scores == 5)

    minority = min([len(s_index[i][0]) for i in range(0, len(s_index))])
    r_scores = []
    r_texts = []

    for i in range(0, len(s_index)):
        sampled_items = np.random.choice(s_index[i][0], minority, replace=False)
        r_scores.extend(scores[sampled_items])
        r_texts.extend(texts[sampled_items])

    return r_texts, r_scores


def get_stratified_test_train(texts, scores):
    X_train, X_test, y_train, y_test = train_test_split(texts, scores, test_size=0.30, random_state=123, stratify=scores)
    return X_train, X_test, y_train, y_test


def save_subsampled_data(source_file, result_name):
    texts, scores = FileHelper.get_formated_messages_from_file(source_file)
    texts, scores = get_undersampled_majorities(texts, scores)
    texts_train, texts_test, scores_train, scores_test = get_stratified_test_train(texts, scores)

    FileHelper.write_message_scores_to_file("data/subsampled/{}_train_{}.json".
                                      format(result_name, len(scores_train)), texts_train, scores_train)
    FileHelper.write_message_scores_to_file("data/subsampled/{}_test_{}.json".
                                      format(result_name, len(scores_test)), texts_test, scores_test)


def build_rates_chart(file, name):
    texts, scores = FileHelper.get_formated_messages_from_file(file)
    scores = np.asarray(scores)
    sizes = [sum(scores == 1), sum(scores == 2), sum(scores == 3), sum(scores == 4), sum(scores == 5)]
    labels = ["Rate 1", "Rate 2", "Rate 3", "Rate 4", "Rate 5"]

    fig, ax1 = plt.subplots()
    ax1.set_title("Rates allocation in category {}".format(name), y=1.08)
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=False, startangle=0)
    ax1.axis('equal')

    fig.savefig("./output/plots/rates/{}RatesAllocation.png".format(name))
    plt.show()

#---------------------Plots-------------------------
build_rates_chart("data/Beauty_5.json", "Beauty")
build_rates_chart("data/Cell_Phones_and_Accessories_5.json", "Phones")
build_rates_chart("data/Grocery_and_Gourmet_Food_5.json", "Grocery")
build_rates_chart("data/Pet_Supplies_5.json", "Pet Supplies")
build_rates_chart("data/Toys_and_Games_5.json", "Games")

#------------------Sumsampling----------------------
# save_subsampled_data("data/Beauty_5.json", "Beauty")
# save_subsampled_data("data/Cell_Phones_and_Accessories_5.json", "Phones")
# save_subsampled_data("data/Grocery_and_Gourmet_Food_5.json", "Grocery")
# save_subsampled_data("data/Pet_Supplies_5.json", "Pet_Supplies")
# save_subsampled_data("data/Toys_and_Games_5.json", "Games")