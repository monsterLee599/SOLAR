#!/usr/bin/env python
# coding: utf-8

# # Feature extraction
# ----------------------------------------
# ## environment requirement
# 
# ### Software
# * Anaconda
# 
# ### Dependence package
# * nltk
# * numpy
# * scipy
# * pandas
# * textstat ( used to analyze readability )
# * pattern ( to find and correct spelling error )
# * textblob ( for sentiment analysis )
# * gensim ( for semantic minning )
# 
# ### Corpus
# * treebank
# * [Harvard General Inquirer](http://www.wjh.harvard.edu/~inquirer/spreadsheet_guide.htm)(used to get uncertainty word and quality word)
# * [MPQA Subjectivity Lexicon](http://mpqa.cs.pitt.edu/lexicons/subj_lexicon/)(used to get subjective word)
# * [Bing Liu's Opinion Lexicon](http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon)(used to get sentiment word)


import numpy, re
import matplotlib
import scipy
import pandas as pd
import json
import csv
import nltk
import textstat
import gensim
import math
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from nltk.tokenize import word_tokenize
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger, tnt
from nltk.tag import DefaultTagger
from nltk.corpus import treebank
# from pattern.en import suggest
from collections import Counter
# from autocorrect import Speller
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from utils import metric
from textblob import TextBlob
from spellchecker import SpellChecker
from sklearn import metrics
from tqdm import tqdm
from dateutil import parser
from nltk.stem import PorterStemmer
from gensim import corpora
from gensim.models import LsiModel
from sklearn.feature_extraction.text import CountVectorizer
import warnings
from statistics import mean
import pickle
# import enchant
# d = enchant.Dict("en_US")
stop_words = stopwords.words('english')
spell = SpellChecker()
warnings.filterwarnings("ignore")
tokenizer = RegexpTokenizer(r'\w+')
cnt = -1


special_words = ['process', 'access', 'less', 'as']
# spell = Speller(lang='en')
class FeatureSummary:
    def __init__(self, str, rating, date, reply):
        self.str = str
        self.rating = int(rating)
        self.date = date
        self.reply = reply
        self.after_extract = self.after_extract()
        if len(self.after_extract) > 1:
            self.str = self.after_extract[1]
        self.processed = self.review_preproesses()
        self.word_list = tokenizer.tokenize(self.processed)
        self.num_of_word = len(self.word_list)
        self.sentence_list = sent_tokenize(self.str)
        self.tagged_text = nltk.pos_tag(self.word_list)
        tag_counter = Counter(tag for word,tag in self.tagged_text)
        self.tag_dict = dict((word, float(count)) for word, count in tag_counter.items())

    def after_extract(self):
        return self.str.split("Full Review")

    def review_preproesses(self):
        global cnt
        cnt = cnt + 1
        text = self.str
        text = text.lower()
        text = text.replace('_', '. ')
        text = re.sub(' +', ' ', text)
        text = text.replace('&#39;', '')
        sentences = [i.strip() for i in text.split('.') if i.strip()]
        for j, sent in enumerate(sentences):
            sentences[j] = self.lemma_word(sent, digit=True)
        return ' . '.join(sentences)

    def lemma_word(self, words, digit=False):
        words = words.replace("'", "")
        word_list = words.split(' ')
        new_word_list = []
        for word in word_list:
            word = WordNetLemmatizer().lemmatize(word, 'v')
            if word not in special_words:
                word = WordNetLemmatizer().lemmatize(word, 'n')
            if digit:
                if word.isdigit():
                    word = '<digit>'
            new_word_list.append(word)
        new_word = ' '.join(new_word_list)
        return new_word

    def find_word(self, df, word):
        count = 0
        word = word.upper()
        for i in df['word']:
            if i == word:
                return count
            count = count + 1
        return None

    def get_review_length(self):
        return len(self.word_list)

    def get_number_of_sentence(self):
        return len(self.sentence_list)

    def get_average_sentence_length(self):
        number_of_sentence = len(self.sentence_list)
        return self.get_review_length() / number_of_sentence

    def get_n_letter_words_number(self, n=0):
        count = 0
        for i in self.word_list:
            if len(i) == n:
                count += 1
        return count

    def get_larger_than_n_letter_words_number(self, n=0):
        count = 0
        for i in self.word_list:
            if len(i) > n:
                count += 1
        return count

    def average_character_of_per_word(self):
        num_of_char = 0
        for i in self.word_list:
            num_of_char += len(i)
        return num_of_char / self.num_of_word

    # def average_word_of_per_paragraph(self):
    #     paragraph_list = self.str.split('\n')
    #     num_of_paragraph = len(paragraph_list[0])
    #     num_of_word = 0
    #     for i in paragraph_list[0]:
    #         num_of_word += len(i)
    #     return num_of_word / num_of_paragraph
    #
    # def number_of_paragargh(self):
    #     paragraph_list = self.str.split('\n')
    #     num_of_paragraph = len(paragraph_list[0])
    #     return num_of_paragraph

    def difficult_words(self):
        return textstat.difficult_words(self.str)

    def spelling_error(self):
        error_count = 0
        for i in self.word_list:
            # if i != spell(i):   #  suggest(i)[0][0]
            # if not d.check(i):
            # corrected = str(TextBlob(i).correct())
            corrected = spell.correction(i)
            if corrected != i:
                error_count += 1
        return error_count/self.num_of_word

    def flesch_reading(self):
        flesch = textstat.flesch_reading_ease(self.str)
        if flesch >=90:
            return 1
        elif flesch >= 80:
            return 2
        elif flesch >= 70:
            return 3
        elif flesch >= 60:
            return 4
        elif flesch >= 50:
            return 5
        elif flesch >= 40:
            return 6
        elif flesch >= 30:
            return 7
        else:
            return 8

    # def coleman_liau_index(self):
    #     return textstat.coleman_liau_index(self.str)

    def dale_chall_re(self):
        dale_score = textstat.dale_chall_readability_score(self.str)
        if dale_score <= 4.9:
            return 1
        elif dale_score<= 5.9:
            return 2
        elif dale_score <= 6.9:
            return 3
        elif dale_score <= 7.9:
            return 4
        elif dale_score <= 8.9:
            return 5
        else:
            return 6

    # def stop_count(self):
    #     english_stops = set(stopwords.words('english'))
    #     stop_word_list = [word for word in self.word_list if word in english_stops]
    #     return len(stop_word_list)

    # def term_frequency(self, all_review):
    #     vectorizer = CountVectorizer()
    #     vectorizer.fit_transform(all_review)
    #     return vectorizer.get_feature_names(),vectorizer.fit_transform(all_review).toarray()
    #
    # def term_frequency_dict(self):
    #     freq = nltk.FreqDist(self.word_list)
    #     return freq

    # noun is N, verb is V, adj is J, determin is DT
    def noun_token(self, tag ="NN"):
        count = 0
        for k, v in self.tag_dict.items():
            if tag in k:
                count += v
        return count

    def verb_token(self, tag="VB"):
        count = 0
        for k, v in self.tag_dict.items():
            if tag in k:
                count += v
        return count

    def adjective_token(self, tag='JJ'):
        count = 0
        for k, v in self.tag_dict.items():
            if tag in k:
                count += v
        return count

    def subjective_token(self, subjective_corpus):
        num_of_sub_word = 0
        subjective_words = [v.lower() for v in subjective_corpus['word'].values]
        types = [1 if w=='strongsubj' else 0.5 for w in subjective_corpus['type'].values]
        for i in self.word_list:
            if i in subjective_words:
                num_of_sub_word = num_of_sub_word + 1
        return num_of_sub_word

    def lex_diversity(self):
        unique_words = []
        for i in self.str:
            if i not in unique_words:
                unique_words.append(i)
        return len(unique_words) / self.num_of_word

    def sentiment_token(self,positive_words,negative_words):
        num_of_sen_word = 0
        for i in self.word_list:
            if i in positive_words or i in negative_words:
                num_of_sen_word = num_of_sen_word + 1
        return num_of_sen_word/self.num_of_word

    # def sentiment(self):
    #     blob = TextBlob(self.str)
    #     return blob.sentiment[0]

    def extremity_calculate(self, mean_rate = 0):
        return abs(self.rating - mean_rate)

    '''
    0 is positive , 1 is netual, and 2 is negative
    '''
    # type 0 is polaity type, type 1 is score
    def polarity_of_review(self,positive_words,negative_words,type=2):
        num_of_pos_word = 0
        num_of_neg_word = 0
        for i in self.word_list:
            if i in positive_words:
                num_of_pos_word = num_of_pos_word + 1
            if i in negative_words:
                num_of_neg_word = num_of_neg_word + 1
        if (num_of_pos_word - num_of_neg_word) / self.num_of_word > 0.02:
            if type == 0:
                return [1,0,0]
            elif type == 1:
                return (num_of_pos_word - num_of_neg_word) / self.num_of_word
            else:
                return 0,(num_of_pos_word - num_of_neg_word) / self.num_of_word
        elif -(num_of_pos_word - num_of_neg_word) / self.num_of_word > 0.015:
            if type == 0:
                return [0,1,0]
            elif type == 1:
                return -(num_of_pos_word - num_of_neg_word) / self.num_of_word
            else:
                return 2,-(num_of_pos_word - num_of_neg_word) / self.num_of_word
        else:
            if type == 0:
                return [0,0,1]
            elif type == 1:
                return abs((num_of_pos_word - num_of_neg_word) / self.num_of_word)
            else:
                return 1,abs((num_of_pos_word - num_of_neg_word) / self.num_of_word)

    def parsing(self,en_stop,p_stemmer):
        stop_word_list = [word for word in self.word_list if not word in en_stop]
        # stemmed_word_list = [p_stemmer.stem(i) for i in stop_word_list]
        return stop_word_list

    def product_quality_related(self,quality_words):
        num_of_quality_word = 0
        for i in self.word_list:
            if i in quality_words:
                num_of_quality_word = num_of_quality_word + 1
        return num_of_quality_word     #/self.num_of_word

    def review_uncertainty(self,if_words):
        num_of_if_word = 0
        for i in self.word_list:
            if i in if_words:
                num_of_if_word = num_of_if_word + 1
        return num_of_if_word    #/ self.num_of_word

    def reply_or_not(self):
        if self.reply == "None":
            return 0
        else:
            return 1

def tf_idf(all_review):
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=1000)
    vectorizer.fit_transform(all_review)
    return vectorizer.get_feature_names(),vectorizer.fit_transform(all_review).toarray()

def helpful_split(labels, n=50, split_score=4):
    # split_score = numpy.percentile(labels, n)
    print("Split threshold is: ", split_score)
    return numpy.asarray([1 if i>=split_score else 0 for i in labels])

def check_helpful_distribution(data):
    from scipy import stats
    import matplotlib.pyplot as plt
    helpful_nums = data['helpful_num'].values
    _, p_value = stats.shapiro(helpful_nums)
    print("P-value is ", p_value)

    prob = data['helpful_num'].value_counts(normalize=True)
    threshold = 0.02
    mask = prob > threshold
    tail_prob = prob.loc[~mask].sum()
    prob = prob.loc[mask]
    prob['>7'] = tail_prob
    print("prob: ", prob)

    prob.plot(kind='bar')
    plt.xticks(rotation=360)
    # plt.tick_params(
    #     axis='x',  # changes apply to the x-axis
        # rotation='horizontal'
        # which='both',  # both major and minor ticks are affected
        # bottom=False,  # ticks along the bottom edge are off
        # top=False,  # ticks along the top edge are off
        # labelbottom=False  # labels along the bottom edge are off
    # )
    plt.show()

def save_array(data):
    with open("../data/input.pkl", "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def save_processed_reviews(reviews):
    with open("../data/processed_reviews.txt", "w") as f:
        f.writelines("%s\n" % r for r in reviews)

def get_feature(data,corpus, subjective_corpus, positive_corpus, negative_corpus, date_split=False, one_app=False):

    if date_split:
        # Isolate the date infomation from author
        date = data['author'].str.extract("([A-Z][^A-Z]*$)")
        data['author']=data['author'].str.replace("([A-Z][^A-Z]*$)","")
        data['date'] = date

    corpus['word'] = corpus['Entry'].str.extract("([A-Z]*)#?([0-9])*")[0]
    corpus['kind'] = corpus['Entry'].str.extract("([A-Z]*)#?([0-9])*")[1]

    # Duplicate review of the same person
    data = data.drop_duplicates('review', keep='last')

    if not one_app:
        app_names = list(data.app_name.unique())
        mean_rates = {}
        for app_name in app_names:
            mean_rates[app_name] = data.loc[data['app_name'] == app_name]['rating'].mean()
    else:
        mean_rate_app = data['rating'].mean()
        print("mean rate is ", mean_rate_app)

    count = 0
    ## Conduct batch size training
    # batch_size = 64
    # list_df = [data[i:i + batch_size] for i in range(0, data.shape[0], batch_size)]

    if not one_app:
        ## Select topest 5000 for true label and lowest 5000 for false label
        # sorted_data = data.sort_values(by=['helpful_num'], ascending=[False])
        data_top = data.loc[data['helpful_num'].astype(int) > 100].sample(n=2500, random_state=1)
        data_low = data.loc[data['helpful_num'].astype(int) < 2].sample(n=2500, random_state=1)
        data_combine = pd.concat([data_top, data_low])
    else:
        data_combine = data

    processed_reviews = []
    feature_vectors = []
    # for idx, list_x in tqdm(enumerate(list_df)):
    for idr, row in tqdm(data_combine.iterrows()):
        # print(row)
        if not one_app:
            help_num = row['helpful_num']
        else:
            help_num = 1
        if not one_app:
            date = row['date']
        else:
            date = row['timestamp']
        rating = row['rating']
        review = row['review']
        # reply = row['reply']
        if not one_app:
            app_name = row['app_name']

        fs = FeatureSummary(review, rating, date, reply=None)
        review_len = fs.get_review_length()
        if review_len == 0:
            continue

        processed_reviews.append(fs.processed)

        sentence_num = fs.get_number_of_sentence()
        avg_sentence_len = fs.get_average_sentence_length()
        char_1_num = fs.get_n_letter_words_number(n=1)
        char_2_num = fs.get_n_letter_words_number(n=2)
        char_more_num = fs.get_larger_than_n_letter_words_number(n=2)
        misspell_num = fs.spelling_error()
        difficult_num = fs.difficult_words()
        flesch_reading = fs.flesch_reading()
        dale_reading = fs.dale_chall_re()

        noun_num = fs.noun_token()
        verb_num = fs.verb_token()
        adj_num = fs.adjective_token()
        subj_num = fs.subjective_token(subjective_corpus)
        lex_diversity = fs.lex_diversity()
        positive_words = positive_corpus[0].values
        negative_words = negative_corpus[0].values
        polarity = fs.polarity_of_review(positive_words=positive_words, negative_words=negative_words, type=1)
        sentiment_word_num = fs.sentiment_token(negative_words=negative_words, positive_words=positive_words)
        if not one_app:
            extremity = fs.extremity_calculate(mean_rate=mean_rates[app_name])
        else:
            extremity = fs.extremity_calculate(mean_rate=mean_rate_app)
        quality_words = corpus[corpus['Quality'].values == 'Quality']['word'].values
        quality_words_low = [w.lower() for w in quality_words]
        quality_word_num = fs.product_quality_related(quality_words=quality_words_low)
        if_words = corpus[corpus['If'].values == 'If']['word'].values
        if_words_low = [w.lower() for w in if_words]
        uncertainty = fs.review_uncertainty(if_words=if_words_low)
        reply_idx = fs.reply_or_not()

        feature_vectors.append(
            [help_num,
             review_len, sentence_num, avg_sentence_len, char_1_num, char_2_num, char_more_num,
             misspell_num, difficult_num, flesch_reading, dale_reading,
             noun_num, verb_num, adj_num, subj_num, lex_diversity,
             polarity, sentiment_word_num, extremity,
             quality_word_num, uncertainty
                # , reply_idx
             ])
        count += 1
        # stylistics: review_len, sentence_num, avg_sentence_len, char_1_num, char_2_num, char_more_num
        # readability: misspell_num, difficult_num, flesch_reading, dale_reading
        # lexicon: noun_num, verb_num, adj_num, lex_diversity, subj_num
        # sentiment: polarity, sentiment_word_num, extremity
        # quality_word_num, uncertainty, reply_idx
        # if count == 500:
        #     break

    word_name_tfidf, tf_idf_mat = tf_idf(processed_reviews)
    feature_arr = numpy.array(feature_vectors)

    print(feature_arr.shape)
    # print(tf_idf_mat.shape)
    feature_arr[:, 1] = pd.cut(feature_arr[:, 1].astype('int8'), 10, labels=False)
    # print(feature_arr[:,0])
    # print(type(feature_arr[:,0]))
    feature_arr[:, 2] = pd.cut(feature_arr[:, 2].astype('int8'), 10, labels=False)
    feature_arr[:, 3] = pd.cut(feature_arr[:, 3].astype('float16'), 10, labels=False)
    feature_arr[:, 4] = pd.cut(feature_arr[:, 4].astype('int8'), 5, labels=False)
    feature_arr[:, 5] = pd.cut(feature_arr[:, 5].astype('int8'), 5, labels=False)
    feature_arr[:, 6] = pd.cut(feature_arr[:, 6].astype('int8'), 10, labels=False)
    feature_arr[:, 7] = pd.cut(feature_arr[:, 7].astype('float16'), 5, labels=False)
    feature_arr[:, 8] = pd.cut(feature_arr[:, 8].astype('int8'), 5, labels=False)
    feature_arr[:, 9] = pd.cut(feature_arr[:, 9].astype('float16'), 5, labels=False)
    feature_arr[:, 10] = pd.cut(feature_arr[:, 10].astype('float16'), 5, labels=False)
    feature_arr[:, 11] = pd.cut(feature_arr[:, 11].astype('float16'), 5, labels=False)
    feature_arr[:, 12] = pd.cut(feature_arr[:, 12].astype('float16'), 5, labels=False)
    feature_arr[:, 13] = pd.cut(feature_arr[:, 13].astype('float16'), 5, labels=False)
    feature_arr[:, 14] = pd.cut(feature_arr[:, 14].astype('float16'), 5, labels=False)
    feature_arr[:, 15] = pd.cut(feature_arr[:, 15].astype('float16'), 5, labels=False)
    feature_arr[:, 16] = pd.cut(feature_arr[:, 16].astype('float16'), 5, labels=False)
    feature_arr[:, 17] = pd.cut(feature_arr[:, 17].astype('float16'), 5, labels=False)
    feature_arr[:, 18] = pd.cut(feature_arr[:, 18].astype('float16'), 5, labels=False)
    y = helpful_split(feature_arr[:, 0].astype('int16'), split_score=2)

    feature_arr = numpy.delete(feature_arr, 0, axis=1)
    feature_arr = numpy.concatenate((feature_arr, tf_idf_mat), axis=1)

    if feature_arr.shape[1] < 1020:
        add_arr = numpy.zeros((feature_arr.shape[0], 1020-feature_arr.shape[1]))
        feature_arr = numpy.concatenate((feature_arr, add_arr), axis=1)
    print(feature_arr.shape)


    return  feature_arr, y # .astype('float16')

    #  Clear duplicated part in review
    # review['review'] = review.apply(review_preproesses,axis=1,after_extract=review["review"].str.extract("(.+?)(Full Review)(.*)").fillna(0))
    #
    # # Corpus preproessing
    # corpus['word'] = corpus['Entry'].str.extract("([A-Z]*)#?([0-9])*")[0]
    # corpus['kind'] = corpus['Entry'].str.extract("([A-Z]*)#?([0-9])*")[1]
    #
    # # Review length(the number of words in review)
    # review_length = review.apply(get_review_length,axis=1)
    #
    # # Number of sentence
    # sentence_length = review.apply(get_number_of_sentence,axis=1)
    #
    # # Average sentence length(the average number of words in sentence)
    # average_sentence_length = review.apply(get_average_sentence_length,axis=1)
    #
    # # Number of 1-letter words
    # number_of_1_letter_words = review.apply(get_n_letter_words_number,axis=1,n=1)
    #
    # # Number of >n-letter words
    # number_of_larger_than_n_letter_words = review.apply(get_larger_than_n_letter_words_number,axis=1,n=7)
    #
    # # Average character of per word
    # avg_char_of_word = review.apply(average_character_of_per_word,axis=1)
    #
    # # Avg. paragraph length
    # avg_paragraph_of_word = review.apply(average_word_of_per_paragraph,axis=1)
    #
    # # Number of paragargh
    # num_of_paragargh = review.apply(number_of_paragargh,axis=1)
    #
    # # Difficule words
    # difficult_count = review.apply(difficult_words,axis=1)
    #
    # # Stop count
    # num_of_stopwords = review.apply(stop_count,axis=1)
    #
    # # Review uncertainty
    # if_words = corpus[corpus['If'].values=='If']['word'].values
    # if_words_low = [w.lower() for w in if_words]
    # review_uncertainty_score = review.apply(review_uncertainty,axis=1,if_words=if_words_low)
    #
    # # Spelling error
    # ## num_of_spelling_error = review.apply(spelling_error,axis=1)
    #
    # # Term frequency
    # word_name_count, term_fre_count = term_frequency(review['review'].values)
    # term_fre = review.apply(term_frequency_dict, axis=1)
    #
    # # Inverse document frequency
    # word_name_tfidf, tf_idf_mat = tf_idf(review['review'].values)
    #
    # # Syntactic token
    # train_sents = treebank.tagged_sents()
    # backoff = DefaultTagger('NN')
    # unitagger = UnigramTagger(train_sents, backoff=backoff)
    # bitagger = BigramTagger(train_sents,backoff=unitagger)
    # tritagger = TrigramTagger(train_sents, backoff=bitagger)
    # tnt_tagger = tnt.TnT(unk=tritagger, Trained=True)
    # tnt_tagger.train(train_sents)
    # syn_count = review.apply(syntanctic_token,axis=1,tagger=tritagger,tag="N")
    #
    # # Subjective Token
    # subjective_words = subjective_corpus['word'].values
    # subjective_token_score = review.apply(subjective_token,axis=1,subjective_words=subjective_words)
    #
    # # Subjective Score
    # subjective_score = review.apply(subjective,axis=1)
    #
    # # Sentiment Token
    # positive_words = postive_corpus[0].values
    # negative_words = negative_corpus[0].values
    # sentiment_token_score = review.apply(sentiment_token,axis=1,positive_words=positive_words,negative_words=negative_words)
    #
    # # Sentiment Score
    # sentiment_score = review.apply(sentiment,axis=1)
    #
    # # Review extremity
    # mean_rating = data['rating'].mean()
    # extremity = data.loc[:,['rating']].apply(extremity_calculate,axis=1,mean_rate=mean_rating)
    #
    # # Polarity
    # positive_words = postive_corpus[0].values
    # negative_words = negative_corpus[0].values
    # polarity = review.apply(polarity_of_review, axis=1, positive_words=positive_words,negative_words=negative_words,type=0)
    # polarity_score = review.apply(polarity_of_review, axis=1, positive_words=positive_words,negative_words=negative_words, type=1)
    # polarity_turtle = review.apply(polarity_of_review, axis=1, positive_words=positive_words,negative_words=negative_words, type=2)
    #
    # # sentiment divergence(depend on polarity calculation)
    # num_of_pos_class = 0
    # pos_force = 0
    # num_of_neg_class = 0
    # neg_force = 0
    # main_stream = (0,0.0)
    #
    # for i,f in polarity_turtle:
    #     if i == 0:
    #         num_of_pos_class = num_of_pos_class + 1
    #         pos_force = pos_force + f
    #     elif i == 2:
    #         num_of_neg_class = num_of_neg_class + 1
    #         neg_force = neg_force + f
    #
    # pos_force = pos_force / num_of_pos_class
    # neg_force = neg_force / num_of_neg_class
    #
    # if num_of_neg_class < num_of_pos_class:
    #     main_stream = (0,pos_force)
    # else:
    #     main_stream = (2, neg_force)
    #
    # sen_diverg = polarity_turtle.apply(sentiment_divergence,main_s = main_stream)
    #
    # # Flesch reading
    # flesh_score = review.apply(flesh_reading,axis=1)
    #
    # # Coleman-Liau Index
    # coleman_score = review.apply(coleman_liau_index,axis=1)
    #
    # # Dale_Chall RE
    # dale_chall_score = review.apply(dale_chall_re,axis=1)
    #
    # # Lex diversity(need to calculate term frequency first)
    # lex_diver = term_fre.apply(lex_diversity)
    #
    # # Rating
    # rating = data['rating']
    #
    # # Age
    # age = data.loc[:,['date']].apply(date_convert,axis=1)
    #
    # # Strength of review sentiment
    # avg_rating = rating.mean()
    # positive_words = postive_corpus[0].values
    # negative_words = negative_corpus[0].values
    # sentiment_strength_score = review.apply(sentiment_strength,axis=1,avg_rating=avg_rating,positive_words=positive_words,negative_words=negative_words)
    #
    # # Semantic information
    # p_stemmer = PorterStemmer()
    # en_stop = set(stopwords.words('english'))
    # parsed_review = review.apply(parsing,axis=1,en_stop=en_stop,p_stemmer=p_stemmer)
    # dictionary = corpora.Dictionary(parsed_review.values)
    # doc_term_matrix = [dictionary.doc2bow(doc) for doc in parsed_review.values]
    #
    # # Product quality relatedness
    # quality_words = corpus[corpus['Quality'].values=='Quality']['word'].values
    # quality_words_low = [w.lower() for w in quality_words]
    # quality_related_score = review.apply(product_quality_related,axis=1,quality_words=quality_words_low)
    #
    # # feature merge
    # feature = [review_length.values]
    # feature = feature + [sentence_length.values]
    # feature = feature + [average_sentence_length.values]
    # feature = feature + [number_of_1_letter_words.values]
    # feature = feature + [number_of_larger_than_n_letter_words.values]
    # feature = feature + [avg_char_of_word.values]
    # # feature = feature + [avg_paragraph_of_word.values]
    # # feature = feature + [num_of_paragargh.values]
    #
    # feature = feature + [difficult_count.values]
    # # feature = feature + [num_of_stopwords.values]
    # feature = feature + [review_uncertainty_score.values]
    # #feature = feature + [num_of_spelling_error.array]
    # feature = feature + [syn_count.values]
    # feature = feature + [subjective_token_score.values]
    # feature = feature + [subjective_score.values]
    # feature = feature + [sentiment_token_score.values]
    # feature = feature + [sentiment_score.values]
    # feature = feature + [extremity.values]
    # feature = feature + [polarity_score.values]
    # feature = feature + [sen_diverg.values]
    # feature = feature + [flesh_score.values]
    # feature = feature + [coleman_score.values]
    # feature = feature + [dale_chall_score.values]
    # feature = feature + [lex_diver.values]
    # feature = feature + [rating.values]
    # feature = feature + [age.values]
    # feature = feature + [sentiment_strength_score.values]
    # feature = feature + [quality_related_score.values]
    #
    # feature = numpy.stack(feature)
    # feature = numpy.concatenate((feature, numpy.array(tf_idf_mat).T))
    # # feature = numpy.concatenate((feature, numpy.array(term_fre_count).T))
    # # feature = numpy.concatenate((feature, numpy.array(doc_term_matrix).T))
    # return feature


# if __name__ == "__main__":
    # data = pd.read_csv('../data/sample.csv')
    # check_helpful_distribution(data)
#     corpus = pd.read_excel('../data/inquirerbasic.xls')
#     subjective_corpus = pd.read_table('../data/subjective_token.txt', sep=" ")
#     positive_corpus = pd.read_table('../data/positive_words.txt', header=None)
#     negative_corpus = pd.read_table('../data/negative_words.txt', header=None)
#
#     corpus['word'] = corpus['Entry'].str.extract("([A-Z]*)#?([0-9])*")[0]
#     corpus['kind'] = corpus['Entry'].str.extract("([A-Z]*)#?([0-9])*")[1]
#     feature = get_feature(data, corpus)

