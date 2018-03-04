import json


class FileHelper:
    @staticmethod
    def get_formated_messages_from_file(file_name="data/Grocery_Gourmet_Food.json", num_items=None):
        texts, summaries, scores = FileHelper.read_data_file(file_name, FileHelper.default_filter, max_num=num_items)
        messages = []
        for i in range(0, len(texts)):
            message = "{0}\n{1}".format(summaries[i].upper(), texts[i])
            messages.append(message)

        return messages, scores

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
    def read_message_scores_from_file(file_name, max_num=None):
        texts = []
        overall = []

        iteration = 0
        with open(file_name, "r") as json_file:
            for line in json_file:
                iteration = iteration + 1
                if max_num is not None and max_num <= iteration:
                    break

                str = json.loads(line)
                review_text = str["text"]
                score = str["overall"]

                texts.append(review_text)
                overall.append(score)

        return texts, overall

    @staticmethod
    def write_message_scores_to_file(file_name, messages, scores):
        with open(file_name, 'w') as f:
            for i, m in enumerate(messages):
                json.dump({'text': m, 'overall': scores[i]}, f, ensure_ascii=False)
                f.write('\n')

