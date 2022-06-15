from __future__ import print_function, division
import os
import torch
import pandas as pd
import chardet
import numpy
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils
from feature_extract import get_feature
from nltk.corpus import words


import warnings
warnings.filterwarnings("ignore")

def data_loading(rootdir, data=None, date_split=False, one_app=False):
    ## Previous data reading
    # data = pd.read_csv(rootdir+'sample.csv')

    ## Current data reading from Mongodb
    if data is None:
        data =  pd.read_pickle(rootdir+'review_mongo.pkl')
    corpus = pd.read_excel(rootdir+'inquirerbasic.xls')
    subjective_corpus = pd.read_table(rootdir+'subjective_token.txt', sep=" ")
    positive_corpus = pd.read_table(rootdir+'positive_words.txt', header=None)
    negative_corpus = pd.read_table(rootdir+'negative_words.txt', header=None)

    feature, y = get_feature(data, corpus, subjective_corpus, positive_corpus, negative_corpus, date_split, one_app)

    return feature, y

class Review_dataset(Dataset):
    def __init__(self, rootdir="../data/", transform=None):
        self.feature, self.y = data_loading(rootdir)   #.T
        # self.id = numpy.random.randint(0, 2, self.__len__())
        self.transform = transform
        print(self.feature.shape)
        print(self.y.shape)

    def __len__(self):
        return len(self.feature)

    def __num__(self):
        return self.feature.shape[1]

    def __getitem__(self, item):
        sample = {'feature': torch.from_numpy(self.feature[item]), 'id': torch.from_numpy(self.y[item])}
        # numpy.array(self.id[item])
        return sample




class ReadMongo():
    def __init__(self):
        pass

    def read_mongo(self):
        from pymongo import MongoClient
        client = MongoClient('localhost', 27017)
        reviews = []
        db = client['googel_review_reply']
        collections = db.list_collection_names()
        for collect in collections:
            cursor = db[collect].find({})
            for doc in cursor:
                del doc['_id']
                doc['app_name'] = collect
                doc['rating'] = int(doc['rating'])
                reviews.append(doc)
        pd_reviews = pd.DataFrame(reviews, columns=['author', 'rating', 'review', 'reply', 'helpful_num', 'app_name'])
        pd_reviews.to_pickle(path="../data/review_mongo.pkl")

class Arminer_dataset():
    def __init__(self, root):
        data = pd.read_table(root, names=["id", "timestamp", "rating", "review"], index_col=0)
        for idx in range(len(data)):
            review = data['review'].iloc[idx]
            review = review[review.find(" ")+1:]
            review = review[review.find(" ")+1:]
            data['review'].iloc[idx] = review
        self.data = data

    def get_data(self):
        return self.data

class CLAP_dataset():
    def __init__(self, root, version_no):
        fp = open(root, 'r', encoding='utf8', errors='ignore')
        lines = fp.readlines()
        data_len = len(lines)
        data = {'review': [], 'rating': [], 'timestamp': []}
        for idx, line in enumerate(lines):
            # line = line.decode('utf-8', 'ignore').encode("utf-8")
            items = line.split("\t")
            if version_no in items[2]:  #.decode("utf-8")
                review = items[-1]
                if self.check_en(review):
                    data['review'].append(review)
                    data['rating'].append(int(items[0]))
                    data['timestamp'].append(int(items[1]))
        self.data = pd.DataFrame.from_dict(data)
        # with open(root, 'rb') as f:
        #     result = chardet.detect(f.read())
        # print("encoding style ", result['encoding'])
        # data = pd.read_table(root, names=["rating", "timestamp", "version", "author", "title", "review"], index_col=0, encoding=result['encoding'])
        # self.data = self.data.loc[self.data['version'] == version_no]
    def get_data(self):
        return self.data

    def check_en(self, review):
        tokens = review.split(' ')
        count_en = 0
        for token in tokens:
            if token in words.words():
                count_en += 1
            if count_en/float(len(tokens)) > 0.3:
                return True
        return False

# if __name__ == "__main__":
    ### Previous data reading
    # data = Review_dataset()
    # print(data[0])

    ### Mongodb data reading
    # reviews = []
    # read_mongo = ReadMongo().read_mongo()

