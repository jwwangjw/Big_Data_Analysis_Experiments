import os
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from rake_nltk import Rake

filepath='C:/Users/lenovo/Desktop/ACL2020'
list1=os.listdir(filepath)
corpus_l=[]
corpus=[]
r = Rake()
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
    txt_name1 = "rake_dic.txt"
    with open(txt_name1, 'a') as file_handle:
        for i in list(r.get_ranked_phrases_with_scores()):
            file_handle.write(i[1]+'\t'+i[0]+'\n')




