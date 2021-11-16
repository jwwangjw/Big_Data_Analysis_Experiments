import string
import os
from collections import defaultdict

import networkx as nx
import numpy as np

from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import nltk
import pandas as pd
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

filepath='C:/Users/lenovo/Desktop/ACL2020'
list1=os.listdir(filepath)
list_words=[]
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
    wordnet_lematizer = nltk.WordNetLemmatizer()
    words = [wordnet_lematizer.lemmatize(raw_word) for raw_word in raw_words]
    # 去除停用词
    filtered_words = [word for word in words if word not in stopwords.words('english')and word not in english_pu and len(word)>2 and word.isdigit()!=True and word[len(word)-1]!=True and word!='cid51']
    filtered_words = pd.Series(filtered_words).str.replace('[^a-zA-Z]', ' ')
    filtered_words = [s.lower() for s in filtered_words]
    print(filtered_words)
     # 连接成字符串，空格分隔
    list_words.append(filtered_words)
doc_frequency=defaultdict(int)
for word_list in list_words:
    for i in word_list:
        doc_frequency[i] += 1
for e in doc_frequency.keys():
    if doc_frequency[e]<2:
        del doc_frequency[e]
print(doc_frequency)

