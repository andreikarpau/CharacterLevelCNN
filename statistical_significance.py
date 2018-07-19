import json
from random import randint


def bootstrap_test_indexes(reviews_count=52442, sample_size=5000, samples_number=20):
    all_indexes = []
    for sn in range(samples_number):
        indexes = []
        for i in range(sample_size):
            indexes.append(randint(0, reviews_count - 1))

        all_indexes.append(indexes)

    with open('analyze/indexes.json', 'w+') as file:
        json.dump(all_indexes, file)


# ---------------------Bootstrap indexes from test dataset-------------------
# bootstrap_test_indexes()
