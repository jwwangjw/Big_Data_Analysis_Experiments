import math
import operator
import string
from collections import defaultdict

from gensim import corpora,models
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
import networkx as nx
import numpy as np
import sys,codecs
import pandas as pd
import numpy as np
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
filepath='C:/Users/lenovo/Desktop/ACL2020'
list1=os.listdir(filepath)
corpus_l=[]
corpus=[]
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
    corpus.append(filtered_words)
def LDA_model(words_list):
    # 构造词典
    # Dictionary()方法遍历所有的文本，为每个不重复的单词分配一个单独的整数ID，同时收集该单词出现次数以及相关的统计信息
    dictionary = corpora.Dictionary(words_list)
    print(dictionary)
    print('打印查看每个单词的id:')
    print(dictionary.token2id)  # 打印查看每个单词的id

    # 将dictionary转化为一个词袋
    # doc2bow()方法将dictionary转化为一个词袋。得到的结果corpus是一个向量的列表，向量的个数就是文档数。
    # 在每个文档向量中都包含一系列元组,元组的形式是（单词 ID，词频）
    corpus = [dictionary.doc2bow(words) for words in words_list]
    print('输出每个文档的向量:')
    print(corpus)  # 输出每个文档的向量

    # LDA主题模型
    # num_topics -- 必须，要生成的主题个数。
    # id2word    -- 必须，LdaModel类要求我们之前的dictionary把id都映射成为字符串。
    # passes     -- 可选，模型遍历语料库的次数。遍历的次数越多，模型越精确。但是对于非常大的语料库，遍历太多次会花费很长的时间。
    lda_model = models.hdpmodel.HdpModel(corpus=corpus,id2word=dictionary)

    return lda_model


lda_model = LDA_model(corpus)
topic_words = lda_model.print_topics(num_topics=2, num_words=5)
print('打印所有主题，每个主题显示5个词:')
print(topic_words)

# 输出该主题的的词及其词的权重
words_list = lda_model.show_topic(0, 50)
print('输出该主题的的词及其词的权重:')
txt_name1="Rp_dic.txt"
with open(txt_name1,'a') as file_handle:
    for i in range(len(words_list)):
        file_handle.write(str(words_list[i][0])+'\t'+str(words_list[i][1]))     # 写入
        file_handle.write('\n')
