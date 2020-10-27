# -*- coding: utf-8 -*-
# file: poems.py
# author: JinTian
# time: 08/03/2017 7:39 PM
# Copyright 2017 JinTian. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
import collections
import os
import sys
from gensim import corpora
from gensim.models.word2vec import Word2Vec
import jieba
import numpy as np

start_token = 'G'
end_token = 'E'


def process_poems(file_name):
    # 诗集
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        for line in f.readlines():
            try:
                title, content = line.strip().split(':')
                content = content.replace(' ', '')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                        start_token in content or end_token in content:
                    continue
                if len(content) < 5 or len(content) > 79:
                    continue
                content = start_token + content + end_token
                poems.append(content)
            except ValueError as e:
                pass
    # 按诗的字数排序
    poems = [[word for word in jieba.cut(words)] for words in poems]
    poems = sorted(poems, key=lambda l: len(line))

    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    # 这里根据包含了每个字对应的频率
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)

    # 取前多少个常用字(可以取字)
    words = words[:len(words)] + (' ',)
    # 每个字映射为一个数字ID,可以用word2rec映射
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(lambda word: word_int_map.get(word, len(words)), poem)) for poem in poems]

    """
    # 第二步：使用语料库建立词典，也就是给预料库中的每个单词标上序号，类似：{'我'：1，'喜欢'：2，'编程'：3,....}首先进行中文分词
    text = [[word for word in jieba.cut(words)] for words in poems]
    dictionary = corpora.Dictionary(text)
    print(dictionary)
    # 第三步，对语料中的每个词进行词频统计,doc2bow是对每一句话进行词频统计，传入的是一个list
    # corpus得到的是一个二维数组[[(0, 1), (1, 1), (2, 1)], [(3, 1), (4, 1)], [(5, 1), (6, 1), (7, 1), (8, 1), (9, 1)]]，意思是编号为0的词出现的频率是1次，编号为2的词出现的频率是1次
    corpus = [dictionary.doc2bow(word) for word in text]
    print(corpus)  # 得到二维数组，最小元素为（词的ID号，词频）
    """
    return poems_vector, word_int_map, words


def process_declare(file_name):
    # 文件
    declare = []
    with open(file_name, "r", encoding='GBK', ) as f:
        for line in f.readlines():
            try:
                content = line.replace(' ', '').replace('|', '')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                        start_token in content or end_token in content:
                    continue
                if len(content) < 1:
                    continue
                content = start_token + content + end_token
                declare.append(content)
            except ValueError as e:
                pass
    # 按诗的字数排序
    declare = [[word for word in jieba.cut(words)] for words in declare]
    declare = sorted(declare, key=lambda l: len(line))

    # 统计每个字出现次数
    all_words = []
    for de in declare:
        all_words += [word for word in de]
    # 这里根据包含了每个字对应的频率
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)

    # 取前多少个常用字(可以取字)
    words = words[:len(words)] + (' ',)
    # 每个字映射为一个数字ID,可以用word2rec映射
    word_int_map = dict(zip(words, range(len(words))))
    declare_vector = [list(map(lambda word: word_int_map.get(word, len(words)), poem)) for poem in declare]

    """
    # 第二步：使用语料库建立词典，也就是给预料库中的每个单词标上序号，类似：{'我'：1，'喜欢'：2，'编程'：3,....}首先进行中文分词
    text = [[word for word in jieba.cut(words)] for words in poems]
    dictionary = corpora.Dictionary(text)
    print(dictionary)
    # 第三步，对语料中的每个词进行词频统计,doc2bow是对每一句话进行词频统计，传入的是一个list
    # corpus得到的是一个二维数组[[(0, 1), (1, 1), (2, 1)], [(3, 1), (4, 1)], [(5, 1), (6, 1), (7, 1), (8, 1), (9, 1)]]，意思是编号为0的词出现的频率是1次，编号为2的词出现的频率是1次
    corpus = [dictionary.doc2bow(word) for word in text]
    print(corpus)  # 得到二维数组，最小元素为（词的ID号，词频）
    """
    return declare_vector, word_int_map, words


def generate_batch(batch_size, data_vec, word_to_int):
    # 每次取64首诗进行训练
    n_chunk = len(data_vec) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        # 起始的位置+batch_size
        end_index = start_index + batch_size

        batches = data_vec[start_index:end_index]
        # 找到这个batch的所有poem中最长的poem的长度,必须要固定长度
        length = max(map(len, batches))
        # 填充一个这么大小的空batch，空的地方放空格对应的index标号
        x_data = np.full((batch_size, length), word_to_int[' '], np.int32)
        for row in range(batch_size):
            # 每一行就是一首诗，在原本的长度上把诗还原上去
            x_data[row, :len(batches[row])] = batches[row]
        y_data = np.copy(x_data)
        # y的话就是x向左边也就是前面移动一个
        y_data[:, :-1] = x_data[:, 1:]
        """
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches
