# coding: utf-8

import glob
import os
import sys
import pickle

from utils import DocumentUtil

main_dirs = ['后缀为.data', '后缀为.idx']
types = ['办事', '互动', '新闻', '政策', '政务']
file_list = []
documents, labels = [], []
doc_util = DocumentUtil()
doc_util.load_word_dict()
for sub_dir in main_dirs:
    for sub_type in types:
        print('%s' % (os.path.join(sub_dir, sub_type, '*')))
        file_list = glob.glob(os.path.join(sub_dir, sub_type, '*'))
        for single_file in file_list:
            try:
                lines = open(single_file, 'r', encoding='utf-8').readlines()
            except:
                lines = None
            if lines:
                documents.append(lines)
                labels.append(sub_type)
print('Loading documents...')
doc_util.load_document(documents, labels)
print('Calculating tfidf...')
_ = doc_util.get_tfidf()
print('Exporting...')
doc_util.export()
