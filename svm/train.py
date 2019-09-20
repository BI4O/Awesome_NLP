import sys
import os
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
import numpy as np
import joblib
from config import tags, get_data_path, get_model_path


def train(train_data, reducer_model_path, classifier_model_path):
    train_data = np.load(train_data)
    train_xs, train_ys = train_data["xs"], train_data["ys"]
    reducer = TruncatedSVD(n_components=300)
    train_xs = reducer.fit_transform(train_xs)
    classifier = SVC(C=10.0, gamma="scale", probability=True)
    classifier.fit(train_xs, train_ys)
    joblib.dump(reducer, reducer_model_path)
    joblib.dump(classifier, classifier_model_path)

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in tags:
        raise ValueError("""usage: 情绪效价，情绪用户""")
    tag_type = sys.argv[1]
    origin_file_path, train_npz_path, test_npz_path = get_data_path(tag_type)
    _, reducer_model_path, classifier_model_path = get_model_path(tag_type)
    train(train_npz_path, reducer_model_path, classifier_model_path)
