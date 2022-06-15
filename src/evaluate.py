# Evaluate the results of topic modeling
import os, json


def evaluate(app_name, topic_sourted):
    get_groundtruth(app_name)

def get_groundtruth(app_name):
    data_dir = "../data/selected_data/metadata/"
    files = os.listdir(data_dir)
    for file in files:
        if app_name in file:
            full_name = file
            break
    metadata_pt = data_dir + full_name
    metadata_fp = open(metadata_pt)
    metadata = json.load(metadata_fp)
    updates = metadata['update_html']
    update_logs = updates.split('<br>')
    new_logs = []
    for update_log in update_logs:

        if update_log.startswith('-'):
            new_log = update_log.strip('-')
            new_log = new_log.strip(' ')
            new_log = new_log.replace('<p>', ' ')
            new_logs.append(new_log)
    print(update_logs)

if __name__ == "__main__":
    evaluate(app_name="ebay", topic_sourted="")
