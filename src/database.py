#!/usr/bin/env python
# -*- coding: utf-8 -*-
# database for saving crawled reviews
# __author__ = "Cuiyun Gao"
# __version__ = "1.0"
# __date__ = "08/12/2017"


import pymongo
from pymongo.errors import DuplicateKeyError, OperationFailure


# define Database class
class Database:
    def __init__(self):
        self.client = pymongo.MongoClient("localhost", 27017)
        print("Connect to mongodb")

    def insert_app_info(self, app_id, app_info):
        # app_info = (app_name, category, ads, price, rating_num, rating_val)
        db = self.client.google_review_helpful_id
        _id = app_id
        insert_info = {"app_name": app_info[0], "category": app_info[1], "ads_info": app_info[2], "price": app_info[3],
                       "rating_num": app_info[4], "rating_val": app_info[5], "_id": _id}
        try:
            db[app_id].insert_one(insert_info)
        except DuplicateKeyError:
            # print("Duplicate key error. The app %s cannot be added." % app_id)
            # print(app_info)
            pass
        except OperationFailure:
            db['.'.join(app_id.split('.')[-3:])].insert_one(insert_info)

    def insert_review_reviews(self, app_id, reviews):
        # review_lists.append((author, rating, review, reply, helpful_num))
        db = self.client.google_review_helpful
        for review in reviews:
            try:
                db[app_id].insert_one(review)
            except DuplicateKeyError:
                pass
                # print("Duplicate key error. The app %s cannot be added." % app_id)
        print("App %s insert review successfully." % app_id)

    def read_review_reply(self):
        review_dict = {}
        db = self.client.googel_review_reply
        collections = db.collection_names()
        for collection in collections:
            review_dict[collection] = []
            for review in db[collection].find():
                review_dict[collection].append(review)
        return review_dict