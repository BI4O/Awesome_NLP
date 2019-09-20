import re
import sys
import jieba
import joblib
import numpy as np
import xlrd
from sklearn.feature_extraction.text import TfidfVectorizer
from config import tags, get_data_path, get_model_path, get_category_names

def text_to_vector(category_names, origin_file_path, vectorizer_model_path, train_npz_path, test_npz_path):
    """
    文本转成数组
    :param origin_file_path: 原始数据文件路径
    :param model_path: 模型路径
    :param train_npz_path: 文本转成数组后【训练数据】保存路径
    :param test_npz_path:  文本转成数组后【测试数据】保存路径
    :return:
    """
    # 正则表达式匹配非中文字符, 用于过滤标点符号和空格
    pattern = re.compile(r"[^\w\u4e00-\u9fff]+")
    jieba.load_userdict("userdict/dict.txt")
    with open("userdict/map.txt", "r", encoding="utf-8") as f:
        word_map = [line.strip().split(" ") for line in f]
    with open("userdict/stop_words.txt", "r", encoding="utf-8") as f:
        stop_words = [line.strip() for line in f]

    xs = []
    ys = []

    wb = xlrd.open_workbook(origin_file_path)
    sheet = wb.sheets()[0]
    for i in range(sheet.nrows):
        text = sheet.cell(i, 1).value.strip()
        text = text.replace("A:", "").replace("B:", "").replace("\n", "")
        text = text.replace("(", "").replace("*", "").replace(")", "").replace(" ", "")
        text = re.sub(pattern, "", text)
        for k, v in word_map:
            text.replace(k, v)

        words = [w for w in jieba.lcut(text) if w not in stop_words]
        xs.append(" ".join(words))

        label = sheet.cell(i, 2).value.strip()
        label = category_names.index(label)
        ys.append(label)

    # vectorizer = CountVectorizer()
    vectorizer = TfidfVectorizer()
    xs = vectorizer.fit_transform(xs).toarray()
    joblib.dump(vectorizer, vectorizer_model_path)

    indices = np.random.permutation(len(xs))
    test_size = int(len(xs) * 0.3)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    train_xs = np.array([xs[i] for i in train_indices])
    train_ys = np.array([ys[i] for i in train_indices])
    test_xs = np.array([xs[i] for i in test_indices])
    test_ys = np.array([ys[i] for i in test_indices])
    np.savez_compressed(train_npz_path, xs=train_xs, ys=train_ys)
    np.savez_compressed(test_npz_path, xs=test_xs, ys=test_ys)

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in tags:
        raise ValueError("""usage: 情绪效价，情绪用户""")
    tag_type = sys.argv[1]
    origin_file_path, train_npz_path, test_npz_path = get_data_path(tag_type)
    vectorizer_model_path, _, _ = get_model_path(tag_type)
    if origin_file_path == '' or vectorizer_model_path == '' or train_npz_path == '' or test_npz_path == '':
        print("路径有点问题......")
    else:
        category_names = get_category_names(tag_type)
        text_to_vector(category_names, origin_file_path, vectorizer_model_path, train_npz_path, test_npz_path)