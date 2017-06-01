# coding: utf-8

import os
import pickle
import jieba

SHOW_INTER = 50

class DocumentUtil(object):

    def __init__(self, stop_words_path='stop_words.txt', document_pkl_path='documents.pkl'):
        self.word2id = None
        self.id2word = None
        self.documents = None
        self.doc_idf = None
        self.labels = {}
        self.stop_words = set()
        self.document_pkl_path = document_pkl_path
        with open(stop_words_path, 'r', encoding='utf-8') as fr:
            while True:
                try:
                    line = fr.readline()
                    if not line:
                        break
                    elif not line.strip():
                        continue
                    else:
                        self.stop_words.add(line.strip())
                except:
                    continue
            self.stop_words |= set([',', '.', '?', '/', '!', ])
        self.load_word_dict()
        self.load_idf()

    def raw_to_document(self, raw_docs):
        if type(raw_docs) == str:
            raw_docs = [raw_docs]
        ret = []
        doc_id = 0
        for raw_doc in raw_docs:
            def is_meaning_word(word):
                if word in self.stop_words:
                    return False
                if all(ord(c) < 128 for c in word):
                    return False
                return True
            #print('raw_doc = %s' % raw_doc)
            #exit(-1)
            doc = []
            for raw_line in raw_doc:
                raw_line = raw_line.strip()
                if all(ord(c) < 128 for c in raw_line):
                    continue
                doc += [x for x in jieba.cut(raw_line) if is_meaning_word(x)]
            ret.append(doc)
            doc_id += 1
            if doc_id % SHOW_INTER == 0:
                print('doc_id = %d' % doc_id)
        return ret

    def load_word_dict(self):
        if os.path.exists('word_dict.pkl'):
            self.word2id = pickle.load(open('word_dict.pkl', 'rb'))
            self.id2word = {}
            for word, word_id in self.word2id.items():
                self.id2word[word_id] = word
        else:
            self.word2id = {}
            self.id2word = {}

    def load_idf(self):
        if os.path.exists('idf.pkl'):
            self.doc_idf = pickle.load(open('idf.pkl', 'rb'))
        else:
            self.doc_idf = {}

    def update_word_dict(self, new_words):
        assert type(new_words) in (list, set)
        if type(new_words) == list:
            new_words = set(new_words)
        for word in new_words:
            if word not in self.word2id.keys():
                word_id = len(self.word2id)
                self.word2id[word] = word_id
                self.id2word[word_id] = word
        pickle.dump(self.word2id, open('word_dict.pkl', 'wb'))

    def load_document(self, documents, labels):
        '''
        document: a set or list of documents, document = list[words]
        '''
        assert type(documents) == list
        try:
            assert documents
        except:
            raise Exception('Documents should not be empty')
        try:
            assert len(documents) == len(labels)
        except:
            raise Exception('len(document) != len(labels)')
        if type(documents) == set:
            documents = list(documents)
        documents = self.raw_to_document(documents)
        self.documents = {}
        doc_id = 0
        for document, label in zip(documents, labels):
            assert type(document) == list
            self.documents[doc_id] = (document, label)
            if label not in self.labels.keys():
                self.labels[label] = len(self.labels) + 1
            doc_id += 1
            if doc_id % SHOW_INTER == 0:
                print('doc_id = %d' % doc_id)
        self.update_word_dict_with_documents()
        pickle.dump((self.documents, self.labels), open(self.document_pkl_path, 'wb'))

    def load_document_from_pkl(self):
        assert self.document_pkl_path
        self.documents, self.labels = pickle.load(open(self.document_pkl_path, 'rb'))
        self.update_word_dict_with_documents()

    def update_word_dict_with_documents(self):
        print('Updating word dict...')
        word_set = set()
        for doc_id, (document, label) in self.documents.items():
            for word in document:
                word_set.add(word)
        self.update_word_dict(word_set)

    def _calc_tfidf(self, documents, train_mode):
        assert self.word2id
        assert self.id2word
        assert self.documents
        # calc inversed document frequency
        print('Calculating idf...')
        if train_mode is True:
            self.doc_idf = {}
            for doc_words in documents:
                for word in set(doc_words):
                    word_id = self.word2id.get(word)
                    self.doc_idf[word_id] = self.doc_idf.get(word_id, 0) + 1
            for word, word_cnt in self.doc_idf.items():
                self.doc_idf[word] = 1.0 / word_cnt
            pickle.dump(self.doc_idf, open('idf.pkl', 'wb'))
        print('Calculating tf...')
        # calc term frequency
        tfidfs = []
        for doc_words in documents:
            total_words = len(doc_words)
            doc_tf = {}
            for word in doc_words:
                if word not in self.word2id.keys():
                    continue
                word_id = self.word2id[word]
                doc_tf[word_id] = doc_tf.get(word_id, 0) + 1.0 / total_words
            # calc tfidf of each words
            tfidf = {}
            for word, tf in doc_tf.items():
                # TODO: check here
                if word not in self.doc_idf.keys():
                    continue
                tfidf[word] = doc_tf.get(word) * self.doc_idf.get(word)
            tfidfs.append(tfidf)
        print('Save tfidf to pickle file...')
        if train_mode is True:
            pickle.dump(tfidfs, open('train_tfidf.pkl', 'wb'))
        else:
            pickle.dump(tfidfs, open('tfidf.pkl', 'wb'))
        return tfidfs
    
    def train_tfidf(self):
        documents = []
        labels = []
        for document, label in self.documents.values():
            documents.append(document)
            labels.append(label)
        tfidfs = self._calc_tfidf(documents, train_mode=True)
        assert len(documents) == len(labels)
        tfidfs = [_ for _ in zip(tfidfs, labels)]
        print('Exporting base tfidf...')
        self.export(tfidfs)

    def get_tfidf(self, documents, labels=None):
        assert documents
        tfidfs = self._calc_tfidf(documents, train_mode=False)
        if labels:
            assert len(documents) == len(labels)
            tfidfs = [_ for _ in zip(tfidfs, labels)]
        return tfidfs

    def export(self, tfidfs):
        # word dict is up to date
        print('labels = %s' % str(self.labels))
        with open('tfidf.txt', 'w', encoding='utf-8') as fw:
            for tfidf, label in tfidfs:
                tfidf_str = ''
                for word_id in range(len(self.word2id)):
                    word_tfidf = tfidf.get(word_id, 0)
                    tfidf_str += '%f,' % word_tfidf
                tfidf_str = tfidf_str[:-1]
                label_str = ('0,' * len(self.labels))[:-1]
                label_str = label_str[:2 * self.labels.get(label)] + '1' + label_str[2 * self.labels.get(label) + 1:]
                line = tfidf_str + '\t' + label_str
                print(line, file=fw)
