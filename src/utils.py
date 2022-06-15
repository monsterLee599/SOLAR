import torch
import torch.nn as nn
import warnings
import feature_extract
# from feature_extract import review_preproesses
warnings.filterwarnings("ignore")

class SVM_loss(nn.Module):
    def __init__(self):
        super(SVM_loss, self).__init__()

    def forward(self, x, y_pred, y, fc_weight, c=0.01):
        print(y_pred)
        loss = torch.mean(torch.clamp(1 - y_pred.t() * y.float(), min=0))  # hinge loss
        loss += c * torch.mean(fc_weight ** 2)  # l2 penalty
        return loss

class metric:
    def __init__(self):
        self.TP = 0
        self.TN = 0
        self.FN = 0
        self.FP = 0

    def clear(self):
        self.TP = 0
        self.TN = 0
        self.FN = 0
        self.FP = 0

    def update(self, predict_id, real_id):
        self.TP += ((predict_id == 1) & (real_id == 1)).sum()
        self.TN += ((predict_id == 0) & (real_id == 0)).sum()
        self.FN += ((predict_id == 0) & (real_id == 1)).sum()
        self.FP += ((predict_id == 1) & (real_id == 0)).sum()

    def precision(self):
        return self.TP / ( self.TP + self.FP)

    def recall(self):
        return self.TP / ( self.TP + self.FN )

    def f_measure(self):
        return 2 * self.precision() * self.recall() / (self.precision() + self.recall())

    def accuracy(self):
        return (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)


def get_preprocessed_review(data):
    data = data.drop_duplicates('review', keep='last')
    print("processed data len ", data.shape[0])
    processed_reviews = []
    for idx, row in data.iterrows():
        review = row['review']
        rating = row['rating']
        date = row['timestamp']
        fs = feature_extract.FeatureSummary(review, rating, date, None)
        processed_reviews.append(fs.word_list)
    return data, processed_reviews
