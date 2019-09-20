import xlrd
import xlsxwriter
import jieba
import re
import joblib
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import get_model_path

def process_data(input_file, vectorizer_model_path):
    # 正则表达式匹配非中文字符, 用于过滤标点符号和空格
    pattern = re.compile(r"[^\w\u4e00-\u9fff]+")
    jieba.load_userdict("userdict/dict.txt")
    with open("userdict/map.txt", "r", encoding="utf-8") as f:
        word_map = [line.strip().split(" ") for line in f]
    with open("userdict/stop_words.txt", "r", encoding="utf-8") as f:
        stop_words = [line.strip() for line in f]

    wb = xlrd.open_workbook(input_file)
    sheet = wb.sheets()[0]
    original_data = []
    xs = []
    for i in tqdm(range(1, sheet.nrows), desc="[1/3] Processing data", dynamic_ncols=True, ascii=True):
        id = str(sheet.cell(i, 0).value)
        text = sheet.cell(i, 1).value.strip()
        original_data.append((id, text))
        
        text = text.replace("A:", "").replace("B:", "").replace("\n", "")
        text = text.replace("(", "").replace("*", "").replace(")", "").replace(" ", "")
        text = re.sub(pattern, "", text)
        for k, v in word_map:
            text.replace(k, v)
        words = [w for w in jieba.lcut(text) if w not in stop_words]
        xs.append(" ".join(words))
    
    print("[2/3] Vectorizing")
    vectorizer = joblib.load(vectorizer_model_path)
    xs = vectorizer.transform(xs).toarray()
        
    return original_data, xs
    
def predict(xs, reducer_model_path, classifier_model_path):
    reducer = joblib.load(reducer_model_path)
    classifier = joblib.load(classifier_model_path)
    xs = reducer.transform(xs)
    prob = []
    with tqdm(total=len(xs), desc="[3/3] Predicting", dynamic_ncols=True, ascii=True) as progressbar:
        while(len(xs) > 0):
            prob += [p[1] for p in classifier.predict_proba(xs[:100])]
            progressbar.update(len(xs[:100]))
            xs = xs[100:]
            
    histogram_stats(prob, "predict_histogram.png")
    
    return prob
            
def histogram_stats(data, filename):
    plt.figure()
    plt.grid(True)
    n, bins, _ = plt.hist(data, bins=20, range=(0, 1))
    plt.savefig(filename)
    for i in range(len(n)):
        print("[%.2f, %.2f) %d" % (bins[i], bins[i + 1], n[i]))
            
def save(original_data, prob, output_file):
    print("Saving results to", output_file)
    with xlsxwriter.Workbook(output_file) as wb:
        sheet = wb.add_worksheet()
        sheet.write(0, 0, "工单编号")
        sheet.write(0, 1, "文本内容")
        sheet.write(0, 2, "置信度")
        for i in range(len(original_data)):
            id, text = original_data[i]
            sheet.write(i + 1, 0, id)
            sheet.write(i + 1, 1, text)
            sheet.write(i + 1, 2, prob[i])

if __name__ == "__main__":
    if((len(sys.argv) < 3) or (sys.argv[1] not in ['情绪效价', '情绪用户'])):
        raise ValueError(""""usage: python predict_excel.py 情绪效价，情绪用户 input_file [output_file]""")
    
    tag_type = sys.argv[1]
    input_file = sys.argv[2]
    if(len(sys.argv) > 3):
        output_file = sys.argv[3]
    else:
        output_file = "predict_output.xlsx"
    
    vectorizer_model_path, reducer_model_path, classifier_model_path = get_model_path(tag_type)
    original_data, xs = process_data(input_file, vectorizer_model_path)
    prob = predict(xs, reducer_model_path, classifier_model_path)
    save(original_data, prob, output_file)