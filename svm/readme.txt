流程:

1. process.py
a. 从Excel文件中读入数据
b. 利用jieba进行分词(加入用户词典, 删去停用词)
c. 用CountVectorizer或TfidfVectorizer生成特征向量
d. 切分数据集(训练集:测试集 = 7:3)
e. 写入文件train.npz, test.npz

2. train.py
a. 读入train.npz中的数据
b. 用SVC算法训练
c. 保存模型model

3. test.py
a. 读入test.npz中的数据
b. 读入模型model
c. 进行预测
d. 打印准确率报告与混淆矩阵

4. predict_excel.py
a. 命令行 python3 predict_excel.py <Excel文件名>, Excel表格首行为表头
b. 读入模型model并进行预测
c. 将文本与置信度写入新的文件out.xls