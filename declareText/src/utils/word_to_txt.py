#!D:\\My Python\\Trans_to_txt.py
# 注意Windows下路径表示
from win32com import client as wc
import os

print('Enter your Director\'s path:')
# mypath = input()
all_FileNum = 0


def Translate(level, path):
    global all_FileNum
    '''
    将一个目录下所有doc文件转成txt
    '''
    # 该目录下所有文件的名字
    files = os.listdir(path)
    for f in files:
        if (not f.endswith('doc') and not f.endswith('docx')):
            continue
        docPath = path + "\\" + f
        # 除去后边的.doc后缀
        if docPath.endswith('.doc'):
            textPath = docPath[:-4]
        else:
            textPath = docPath[:-5]
        pathstr = textPath.split("\\")
        textPath = textPath.replace(pathstr[len(pathstr) - 1], "wordtotxt\\" + pathstr[len(pathstr) - 1])
        # 改成txt格式
        word = wc.Dispatch('Word.Application')
        doc = word.Documents.Open(docPath)
        doc.SaveAs(textPath + '.txt', 2)
        doc.Close()
        all_FileNum = all_FileNum + 1


if __name__ == '__main__':
    mypath = "F:\python_space\declareText\src\dataset\data"
    Translate(1, mypath)
    print('文件总数 = ', all_FileNum)
