# coding: utf-8

import glob
import os
import sys
import pickle

from accessory import DatasetTools
from utils import DocumentUtil

document_pkl_file = 'documents.pkl'

doc_util = DocumentUtil()
if os.path.exists(document_pkl_file):
    doc_util.load_document_from_pkl()
else:
    main_dirs = ['后缀为.data', '后缀为.idx']
    types = ['办事', '互动', '新闻', '政策', '政务']
    file_list = []
    documents, labels = [], []

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
print('Training tfidf...')
_ = doc_util.train_tfidf()
print('Exporting base tfidf...')
doc_util.export()

print('Calculating dataset tfidf...')
dataset = [_ for _ in zip(doc_util.documents.items())]
dataset_tool = DatasetTools()
dataset_tool.load_dataset(dataset)
train, valid, test = dataset_tool.fold_split([7,2,1])
train = doc_util.get_tfidf([_ for _ in zip(*train)])
valid = doc_util.get_tfidf([_ for _ in zip(*valid)])
test = doc_util.get_tfidf([_ for _ in zip(*test)])
pickle.dump(train, open(os.path.join('dataset', 'train.pkl'), 'wb'))
pickle.dump(valid, open(os.path.join('dataset', 'valid.pkl'), 'wb'))
pickle.dump(test, open(os.path.join('dataset', 'test.pkl'), 'wb'))
