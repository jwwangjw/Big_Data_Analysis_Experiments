import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')  # 忽略警告
import sys, codecs
import string
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
import pandas as pd
import numpy as np
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


def getWordVecs(wordList, model):
    name = []
    vecs = []
    for word in wordList:
        word = word.replace('\n', '')
        try:
            if word in model:  # 模型中存在该词的向量表示
                name.append(word.encode('utf8'))
                vecs.append(model[word])
        except KeyError:
            continue
    a = pd.DataFrame(name, columns=['word'])
    b = pd.DataFrame(np.array(vecs, dtype='float'))
    return pd.concat([a, b], axis=1)


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
    filtered_words = [word for word in words if word not in stopwords.words('english')and word not in english_pu and len(word)>2 and word.isdigit()!=True and word[len(word)-1]!=True]
    print(list(set(filtered_words)))
     # 连接成字符串，空格分隔
    list_words.append(filtered_words)