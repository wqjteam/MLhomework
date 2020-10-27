import os
import codecs
import pandas as pd
import jieba
import jieba.analyse

'''定义变量
文件路径/文件内容/关键字（5个）'''
filepaths = []
contents = []
tag1 = []
tag2 = []
tag3 = []
tag4 = []
tag5 = []


def fetchKW(file_path):
    # 遍历文件，同时得到关键字
    for root, dirs, files in os.walk(r'path'):
        for name in files:
            filepath = root + '\\' + name  # 根目录加文件名构成文件路径
            f = codecs.open(filepath, 'r', 'utf-8')  # 根据文件路径以只读的形式打开文件
            content = f.read().strip()  # 将文件内容传入content变量
            f.close()  # 关闭文件
            tags = jieba.analyse.extract_tags(content, topK=5)  # 根据文件内容获取前5个关键字(出现次数最多)
            filepaths.append(filepath)  # 得到文件路径的集合
            contents.append(content)  # 得到文件内容的集合
            tag1.append(tags[0])
            tag2.append(tags[1])
            tag3.append(tags[2])
            tag4.append(tags[3])
            tag5.append(tags[4])

    tagDF = pd.DataFrame({
        '文件路径': filepaths,
        '文件内容': contents,
        '关键词1': tag1,
        '关键词2': tag2,
        '关键词3': tag3,
        '关键词4': tag4,
        '关键词5': tag5})
    return tagDF
