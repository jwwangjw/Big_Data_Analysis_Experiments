import math
import operator
import string
from collections import defaultdict

from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.converter import TextConverter, PDFPageAggregator
from pdfminer.layout import LAParams
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfdevice import PDFDevice
from pdfminer.pdfpage import PDFPage
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os
import sys,codecs
import pandas as pd
import numpy as np
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
filepath='C:/Users/lenovo/Desktop/ACL2020'
list1=os.listdir(filepath)
list_words=[]
for i in range(len(list1)):
    outs=""
    fp = open(filepath+'/'+list1[i], 'rb')
    parser = PDFParser(fp)
    doc = PDFDocument(parser=parser)
    parser.set_document(doc=doc)
    resource = PDFResourceManager()
    laparam = LAParams()
    device = PDFPageAggregator(resource, laparams=laparam)
    interpreter = PDFPageInterpreter(resource, device)
    for page in PDFPage.get_pages(fp):
        interpreter.process_page(page)
        layout = device.get_result()
        for out in layout:
            if hasattr(out, 'get_text'):
                outs=out.get_text()+outs
    outs=outs.lower().replace('\n','')
    english_pu=['’','“','“']
    punctuation_map = dict((ord(char), None) for char in string.punctuation)
    without_punctuation = outs.translate(punctuation_map)  # 去除文章标点符号
    raw_words = nltk.word_tokenize(without_punctuation)  # 将文章进行分词处理,将一段话转变成一个list
    wordnet_lematizer = WordNetLemmatizer()
    words = [wordnet_lematizer.lemmatize(raw_word) for raw_word in raw_words]
    # 去除停用词
    filtered_words = [word for word in words if word not in stopwords.words('english')and word not in english_pu and len(word)>2 and word.isdigit()!=True and word[len(word)-1]!=True and word!='cid51']
    print(filtered_words)
     # 连接成字符串，空格分隔
    list_words.append(filtered_words)

doc_frequency=defaultdict(int)
for word_list in list_words:
    for i in word_list:
        doc_frequency[i] += 1

# 计算每个词的TF值
word_tf = {}  # 存储没个词的tf值
for i in doc_frequency:
    word_tf[i] = doc_frequency[i] / sum(doc_frequency.values())

# 计算每个词的IDF值
doc_num = len(list_words)
word_idf = {}  # 存储每个词的idf值
word_doc = defaultdict(int)  # 存储包含该词的文档数
for i in doc_frequency:
    for j in list_words:
        if i in j:
            word_doc[i] += 1
for i in doc_frequency:
    word_idf[i] = math.log(doc_num / (word_doc[i] + 1))

# 计算每个词的TF*IDF的值
word_tf_idf = {}
for i in doc_frequency:
    word_tf_idf[i] = word_tf[i] * word_idf[i]

# 对字典按值由大到小排序
dict_feature_select = sorted(word_tf_idf.items(), key=operator.itemgetter(1), reverse=True)
txt_name1="tf-idf_dic.txt"
print(dict_feature_select[1][0])
with open(txt_name1,'a') as file_handle:
    for i in range(50):
        file_handle.write(str(dict_feature_select[i][0])+'\t'+str(dict_feature_select[i][1]))     # 写入
        file_handle.write('\n')





