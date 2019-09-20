import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from config import tags, get_model_path, get_data_path, get_category_names


def test(category_names, reducer_model_path, classifier_model_path, test_npz_path):
    reducer = joblib.load(reducer_model_path)
    classifier = joblib.load(classifier_model_path)
    test_data = np.load(test_npz_path)
    test_xs, test_ys = test_data["xs"], test_data["ys"]

    test_xs = reducer.transform(test_xs)
    pred = classifier.predict(test_xs)
    print_report(pred, test_ys, category_names)

    # Manual threshold
    prob = [p[1] for p in classifier.predict_proba(test_xs)]
    threshold = 0.8
    pred = [1 if (p >= threshold) else 0 for p in prob]
    print("====================")
    print("Threshold:", threshold)
    print_report(pred, test_ys, category_names)
    histogram_stats(prob, "test_histogram.png")


def print_report(y_pred, y_true, category_names=None):
    print("Classification Report")
    print(classification_report(y_true, y_pred, target_names=category_names))
    print("")
    print("Confusion Matrix")
    print(confusion_matrix(y_true, y_pred))
    print("")


def histogram_stats(data, filename):
    plt.figure()
    plt.grid(True)
    n, bins, _ = plt.hist(data, bins=20, range=(0, 1))
    plt.savefig(filename)
    for i in range(len(n)):
        print("[%.2f, %.2f) %d" % (bins[i], bins[i + 1], n[i]))


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in tags:
        raise ValueError("""usage: 情绪效价，情绪用户""")
    tag_type = sys.argv[1]
    print('当前测试标签：' + tag_type)
    origin_file_path, train_npz_path, test_npz_path = get_data_path(tag_type)
    _, reducer_model_path, classifier_model_path = get_model_path(tag_type)
    print("====================")
    print("路径:")
    print("reducer_model_path = ", reducer_model_path)
    print("classifier_model_path = ", classifier_model_path)
    print("test_npz_path = ", test_npz_path)
    print("====================")
    category_names = get_category_names(tag_type)
    test(category_names, reducer_model_path, classifier_model_path, test_npz_path)
