# coding: utf-8

import random

random.seed(13061156)

class DatasetTools(object):
    def __init__(self):
        self.dataset = None

    def load_dataset(self, dataset):
        assert type(dataset) == list
        self.dataset = dataset

    def fold_split(self, partitions):
        assert self.dataset
        assert type(partitions) == list
        all_partitions = 1.0 * sum(partitions)
        partitions = [x / all_partitions for x in partitions]
        threshold, top = [], 0
        for partition in partitions:
            top += partition
            threshold.append(top)
        # print('threshold = %s' % threshold)
        folds = tuple([[] for _ in range(3)])
        for data in self.dataset:
            p = random.random()
            for pos, sum_p in enumerate(threshold):
                if p < sum_p:
                    folds[pos].append(data)
                    break
        len_folds = [len(fold) for fold in folds]
        # print('len_folds = %s' % len_folds)
        return folds
