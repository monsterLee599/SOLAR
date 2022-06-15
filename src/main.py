import argparse
import numpy as np
import torch
import torch.optim as optim
from model import LinearSVM
import dataset
import xlsxwriter
import os.path
import trainer
from utils import SVM_loss, metric
import warnings
warnings.filterwarnings("ignore")
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# import weka.core.jvm as jvm
from utils import get_preprocessed_review
import pandas as pd
import nltk
nltk.data.path.append('/data/cygao/nltk_data/')
from nltk.corpus import stopwords
from torch.utils.data import TensorDataset, DataLoader
from gensim import corpora
import numpy as np
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from bjst import BJST
from sklearn import metrics
import pickle
import logging

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)
device = "cpu"

my_stoplst = ["app", "good", "excellent", "awesome", "please", "they", "very", "too", "like", "love", "nice", "yeah",
"amazing", "lovely", "perfect", "much", "bad", "best", "yup", "suck", "super", "thank", "great", "really",
"omg", "gud", "yes", "cool", "fine", "hello", "alright", "poor", "plz", "pls", "google", "facebook",
"three", "ones", "one", "two", "five", "four", "old", "new", "asap", "version", "times", "update", "star", "first",
"rid", "bit", "annoying", "beautiful", "dear", "master", "evernote", "per", "line", "oh", "ah", "cannot", "doesnt",
"won't", "dont", "unless", "you're", "aren't", "i'd", "can't", "wouldn't", "around", "i've", "i'll", "gonna", "ago",
"you'll", "you'd", "28th", "gen", "it'll", "vice", "would've", "wasn't", "year", "boy", "they'd", "isnt", "1st", "i'm",
"nobody", "youtube", "isn't", "don't", "2016", "2017", "since", "near", "god"]

def SVM_training(train_x, train_y, test_x, true_y):
    clf = svm.SVC(probability=True)
    clf.fit(train_x, train_y)
    predicted_y = clf.predict(test_x)
    predict_y_prob = clf.predict_proba(test_x)
    precision = metrics.precision_score(true_y, predicted_y)
    recall = metrics.recall_score(true_y, predicted_y)
    f_measure = 2 * precision * recall / (precision + recall)
    print("Recall: ", precision, " Precision: ", recall, " F-score: ", f_measure)
    compute_metric(true_y, predicted_y, predict_y_prob)
    save_fn = "../data/svm_model_2.pkl"
    pickle.dump(clf, open(save_fn, "wb"))
    # loaded_model = pickle.load(open(save_fn, "rb"))
    # result = loaded_model.predict_proba(test_x)
    # print(result)

def RF_training(train_x, train_y, test_x, true_y):
    clf = RandomForestClassifier(max_depth=3, random_state=0)
    clf.fit(train_x, train_y)
    predicted_y = clf.predict(test_x)
    predict_y_prob = clf.predict_proba(test_x)
    precision = metrics.precision_score(true_y, predicted_y)
    recall = metrics.recall_score(true_y, predicted_y)
    f_measure = 2 * precision * recall / (precision + recall)
    print("Recall: ", precision, " Precision: ", recall, " F-score: ", f_measure)
    compute_metric(true_y, predicted_y, predict_y_prob)
    # met = metric()
    # met.update(predicted_y, true_y)
    # print("Recall: ", met.recall(), " Precision: ", met.precision(), " F-score: ", met.f_measure())
    save_fn = "../data/rf_model_2.pkl"
    pickle.dump(clf, open(save_fn, "wb"))
    # loaded_model = pickle.load(open(save_fn, "rb"))
    # result = loaded_model.predict_proba(test_x)
    # print(result)

def MNB_training(train_x, train_y, test_x, true_y):
    clf = GaussianNB()
    train_x = train_x.astype('float16')
    test_x = test_x.astype('float16')
    clf.fit(train_x, train_y)
    predicted_y = clf.predict(test_x)
    predict_y_prob = clf.predict_proba(test_x)
    precision = metrics.precision_score(true_y, predicted_y)
    recall = metrics.recall_score(true_y, predicted_y)
    f_measure = 2 * precision * recall / (precision + recall)
    print("Recall: ", precision, " Precision: ", recall, " F-score: ", f_measure)
    compute_metric(true_y, predicted_y, predict_y_prob)
    save_fn = "../data/nb_model_2.pkl"
    pickle.dump(clf, open(save_fn, "wb"))
    # loaded_model = pickle.load(open(save_fn, "rb"))
    # result = loaded_model.predict_proba(test_x)
    # print(result)

def compute_metric(true_y, predicted_y, probs_y):
    tp = tn = fp = fn = 0
    for id_y, y in enumerate(true_y):
        if y == 1:
            if predicted_y[id_y] == y:
                tp += 1
            else:
                tn += 1
        else:
            if predicted_y[id_y] == y:
                fn += 1
            else:
                fp += 1
    hp = tp/float(sum(predicted_y))
    hr = tp/float(sum(true_y))
    up = fn/float(len(predicted_y)-sum(predicted_y))
    ur = fn/float((len(predicted_y)-sum(true_y)))
    print("Helpful precision: ", hp, " recall: ", hr, " f-score: ", 2*hp*hr/(hp+hr))
    print("Unhelpful precision: ", up, " recall: ", ur, " f-score: ", 2*up*ur/(up+ur))
    probs_y_extracted = np.array([probs_y[idy][y] for idy, y in enumerate(true_y)])
    # print(probs_y_extracted)
    # print(true_y)
    print(probs_y_extracted.shape, true_y.shape)
    fpr, tpr, thresholds = metrics.roc_curve(true_y, probs_y_extracted, pos_label=2)
    auc = metrics.auc(fpr, tpr)
    roc_auc = metrics.roc_auc_score(true_y, probs_y_extracted)
    print("ROC AUC is ", roc_auc, "; AUC score is ", auc)


def main(args):
    # review_dataset = dataset.Review_dataset()
    # train_loader = torch.utils.data.DataLoader(review_dataset, batch_size=64, shuffle=True, num_workers=4)
    # test_loader = torch.utils.data.DataLoader(dataset.Review_dataset(), batch_size=64, shuffle=True, num_workers=4)
    # model = LinearSVM(review_dataset.__num__(),2)
    # model.to(device)
    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    # trainer.train(train_loader, train_loader, model, SVM_loss(), optimizer, 20)


    #### New feature extraction part in our TR paper
    # feature, y = dataset.data_loading(rootdir="../data/", date_split=True, one_app=False)
    # feature_train, feature_test, train_y, test_y = train_test_split(feature, y, test_size = 0.2, random_state = 42)
    #
    # SVM_training(feature_train, train_y, feature_test, test_y)
    # RF_training(feature_train, train_y, feature_test, test_y)
    # MNB_training(feature_train, train_y, feature_test, test_y)


    #### BST component
    # data = pd.read_csv('../data/sample.csv')
    # reviews = get_preprocessed_review(data)

    # Build dict
    # bjst_dictionary = build_dict(reviews)

    # BJST
    # phi, theta = bjst(reviews, bjst_dictionary)
    n_topics = 8
    n_iters = 1000
    n_top_reviews = 8
    filter = False
    app_name = "barebones"
    # data = dataset.Arminer_dataset("../data/datasets/swi  ftkey/swiftkey-case.txt").get_data()
    # data = dataset.CLAP_dataset("../data/selected_data/reviews/com.ebay.mobile/com.ebay.mobile_2.txt", "2.6").get_data()
    # data = dataset.CLAP_dataset("../data/selected_data/reviews/com.alfray.timeriffic/com.alfray.timeriffic_2.txt", "1.10").get_data()
    data = dataset.CLAP_dataset("../data/selected_data/reviews/acr.browser.barebones/acr.browser.barebones_2.txt", "3.0").get_data()
    # data = dataset.CLAP_dataset("../data/selected_data/reviews/air.hmbtned/air.hmbtned_2.txt", "3.0").get_data()
    processed_data, reviews = get_preprocessed_review(data)

    if filter:
        feature, y = dataset.data_loading(rootdir="../data/", data=processed_data, date_split=False, one_app=True)
        loaded_model = pickle.load(open("../data/svm_model_2.pkl", "rb"))
        result = loaded_model.predict_proba(feature)
        n_doc = result.shape[0]
        helpful_ids = []
        helpful_reviews = []
        for id_result in range(n_doc):
            if result[id_result][0]<result[id_result][1]:
                helpful_ids.append(id_result)
                helpful_reviews.append(reviews[id_result])
        helpful_data = processed_data.iloc[helpful_ids]
        print('Helpful review number is ', len(helpful_ids))
    else:
        helpful_reviews = reviews
        helpful_data = processed_data

    # Build dict
    bjst_dictionary = build_dict(helpful_reviews)

    # BJST
    phi, theta = bjst(helpful_reviews, bjst_dictionary, n_topics, n_iters)

    #  w[0] * topic_proportion + w[1] * topic_neg_score + w[2] * avg_rating + w[3] * time_score
    topic_score, topic_id_matrix = topic_ranking(helpful_data, phi, theta, [0.3, 0.3, 0.1, 0.3])
    print("topic score ", topic_score)
    # w[0] * rating_score  + w[1] * time_score + w[2] * polarity_score[:,0] + w[3] * polarity_score[:,1] + w[4] * polarity_score[:,2] + w[5] * proportion_score + w[6] * word_score
    data_scored, review_score = review_ranking(helpful_data, topic_id_matrix, phi, theta, [0.1, 0.1, 0.2, 0.09, 0.01, 0.2, 0.3])
    print('review score ', review_score)
    print(review_score.shape)
    target_topic = topic_score.argmax()
    # print(target_topic)
    target_review = data_scored[topic_id_matrix== target_topic]
    topic_sorted = np.argsort(topic_score)[::-1]
    print('topic associated review len ', target_review.shape[0])
    # review_topids = np.argsort(target_review["score"].values)[::-1][:n_top_reviews]
    # print('review topids ', review_topids)
    # print(target_review.iloc[review_topids]["review"])

    # select the top reviews of each topic
    save_xlsx_path = "../data/datasets/nofiltering_%s_topic_%s.xlsx" % (app_name, str(n_topics))
    if not os.path.exists(save_xlsx_path):
        workbook = xlsxwriter.Workbook(save_xlsx_path)
        workbook.close()

    for id_topic in topic_sorted:
        target_review = data_scored[topic_id_matrix== id_topic]
        target_topids = np.argsort(target_review["score"].values)[::-1][:n_top_reviews]
        top_reviews = target_review.iloc[target_topids]  #["review"]
        print("for topic ", id_topic, ", the top reviews are ")
        print(top_reviews)

        with pd.ExcelWriter(save_xlsx_path, engine="openpyxl", mode='a') as fw:
            top_reviews.to_excel(fw, sheet_name=str(id_topic))
    return topic_sorted

def train_SVM(X, Y, model, args):
    X = torch.Tensor(X)
    Y = torch.Tensor(Y)
    N = len(Y)
    # total_data = TensorDataset(X, Y)
    # train_loader = torch.utils.data.DataLoader(total_data, batch_size=args.batchsize, shuffle=True, num_workers=args.worker_num)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    model.train()

    for epoch in range(args.epoch):
        perm = torch.randperm(N)
        sum_loss = 0

        for i in range(0, N, args.batchsize):
            x = X[perm[i : i + args.batchsize]].to(args.device)
            y = Y[perm[i : i + args.batchsize]].to(args.device)

            optimizer.zero_grad()
            output = model(x).squeeze()
            weight = model.weight.squeeze()

            loss = torch.mean(torch.clamp(1 - y * output, min=0))
            loss += args.c * (weight.t() @ weight) / 2.0

            loss.backward()
            optimizer.step()

            sum_loss += float(loss)
        print("Epoch: {:4d}\tloss: {}".format(epoch, sum_loss / N))

def build_dict(dict_input):
    stoplist = stopwords.words('english') + my_stoplst
    dictionary = corpora.Dictionary(dict_input)
    dictionary.filter_tokens(map(dictionary.token2id.get, stoplist))
    dictionary.compactify()
    dictionary.filter_extremes(no_below=2, keep_n=None)
    dictionary.compactify()
    return dictionary

def load_senti_lex(fn):
    senti_lex = {}
    with open(fn) as fin:
        lines = fin.readlines()
        for line in lines:
            terms = line.strip().split("\t")
            senti_lex[terms[0]] = int(terms[1])
    return senti_lex

def bjst(input, dictionary, n_topics=8, n_iter=500):
    X = []
    X_d = []
    # build btm input
    for k, text in enumerate(input):
        X_d_1 = []
        for i, tx_i in enumerate(text):
            w_tx_i = dictionary.token2id.get(tx_i)
            if not w_tx_i:
                continue
            for j, tx_j in enumerate(text[i + 1: i + 18]):
                w_tx_j = dictionary.token2id.get(tx_j)
                if w_tx_j:
                    X.append((w_tx_i, w_tx_j))
                    X_d_1.append((w_tx_i, w_tx_j))
        X_d.append(X_d_1)
    # load sentiment polarity dict
    logging.info("loading senti lexicon")
    senti_lex = load_senti_lex("../data/top_polar_words.txt")
    senti_lex_n = {}
    shallow_dict = {}  # labeling phrase
    for w, wid in dictionary.token2id.items():
        if "_" in w:
            ws = w.split("_")
            for wsi in ws:
                if wsi not in shallow_dict:
                    shallow_dict[wsi] = []
                shallow_dict[wsi].append(wid)
    for word, value in senti_lex.items():
        wid = dictionary.token2id.get(word)
        swids = shallow_dict.get(word)
        if wid:
            senti_lex_n[wid] = value
        if swids:
            for swid in swids:
                senti_lex_n[swid] = value

    model = BJST(n_topics=n_topics, vocab_size=len(dictionary), n_senti=3, senti_lex=senti_lex_n, n_iter=n_iter,
                  refresh=50)
    model.fit(X)
    phi = model.phi_lzw     # sentiment, topic, word
    theta = model.compute_theta_dlz(X_d)    # document, sentiment, topic
    return phi, theta


def topic_ranking(review, phi, theta, w=[0.25, 0.25, 0.25, 0.25]):
    print("review len ", len(review))
    print("original theta shape ", theta.shape)
    theta[np.isnan(theta)] = 1 / theta.shape[2]
    print("removed nan theta shape ", theta.shape)
    n_doc, n_senti, n_topic = theta.shape
    topic_matrix = np.zeros((n_doc, n_topic))
    topic_max_ids = np.zeros(n_doc)
    # cluster
    for n_row in range(n_doc):
        topic_matrix[n_row] = np.sum(theta[n_row], axis=0)
        topic_max_ids[n_row] = np.argmax(topic_matrix[n_row])
    topic_max_ids = topic_max_ids.astype(int)
    print(topic_matrix.shape)
    print("topic_matrix ", topic_matrix)

    # topic proportion
    topic_proportion = theta.sum(0).sum(0)
    topic_proportion /= max(topic_proportion)

    # topic negative score
    topic_neg_score = theta[:, 0, :].sum(0)
    topic_neg_score /= max(topic_neg_score)

    # average rating and time
    avg_rating = np.zeros((theta.shape[2], 2))
    time_score = np.zeros((theta.shape[2], 2))
    for idx in range(n_doc):
        rating = review["rating"].iloc[idx]
        topic_idx = topic_max_ids[idx]
        timestamp = review["timestamp"].iloc[idx] - review["timestamp"].min()
        avg_rating[topic_idx][0] += rating
        avg_rating[topic_idx][1] += 1
        time_score[topic_idx][0] += timestamp
        time_score[topic_idx][1] += 1
    print("time_score shape ", time_score.shape)
    print(time_score)
    print(avg_rating)
    avg_rating = (max(review["rating"])-avg_rating[:, 0] / avg_rating[:, 1]).astype(float) / max(review["rating"])
    time_score = time_score[:, 0] / time_score[:, 1] / (max(review["timestamp"]) - review["timestamp"].min())

    topic_score = w[0] * topic_proportion + w[1] * topic_neg_score + w[2] * avg_rating + w[3] * time_score
    return topic_score, topic_max_ids


def review_ranking(data, topic_max_ids, phi, theta, w=[0.2, 0.2, 0.1, 0.05, 0.05, 0.2, 0.2]):
    n_doc = data.shape[0]
    # rating
    rating_score = (max(data["rating"].values) - data["rating"].values) / max(data["rating"].values)

    # time
    time_score = (data["timestamp"].values - min(data["timestamp"].values)) / (
                max(data["timestamp"].values) - min(data["timestamp"].values))

    # polarity
    polarity_score = np.zeros((n_doc, 3))
    for idx in range(n_doc):
        polarity_score[idx] = theta[idx, :, topic_max_ids[idx]]

    # proportion
    proportion_score = np.zeros(n_doc)
    review_topic_p = theta.sum(1)
    for idx in range(len(data)):
        proportion_score[idx] = review_topic_p[idx, topic_max_ids[idx]]

    # number of word
    word_score = np.zeros(n_doc)
    for idx in range(n_doc):
        word_len = len(data["review"].iloc[idx].split())
        word_score[idx] = word_len
    word_score /= float(max(word_score))

    data["score"] = w[0] * rating_score + w[1] * time_score + w[2] * polarity_score[:, 0] + w[3] * polarity_score[:,
                                                                                                   1] + w[
                        4] * polarity_score[:, 2] + w[5] * proportion_score + w[6] * word_score
    return data, data["score"].values

def read_input(input_fn, review_fn):
    with open(input_fn, "rb") as fr1:
        features = pickle.load(fr1)
    print(type(features))

    with open(review_fn) as fr2:
        processed_reviews = fr2.readlines()
    processed_reviews = [i.strip("\n") for i in processed_reviews]
    return features, processed_reviews

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out_size", type=int, default=2)
    parser.add_argument("--worker_num", type=int, default=4)
    args = parser.parse_args()

    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)

    main(args)
    # feature, y = dataset.data_loading(rootdir="../data/")
    # feature_size = feature.shape[1]
    # model = torch.nn.Linear(feature_size, args.out_size)
    # model.to(args.device)
    #
    # train_SVM(feature, y, model, args)
