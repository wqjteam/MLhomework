import glob
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = curPath[:curPath.find("ultimatemlwork\\") + len("ultimatemlwork\\")]

maxepoch = ""
for f in os.listdir(rootPath + "\\yolov5\\runs\\detect\\"):
    fp = f
    max = -1
    if ((int(fp.replace("exp", ""))) > max):
        max = int(fp.replace("exp", ""))
        maxepoch = f
label_path = rootPath + "\\yolov5\\runs\\detect\\" + maxepoch
jpg_path = rootPath + "\\yolov5\\runs\\detect\\" + maxepoch
result_file = glob.glob(label_path + '*.txt')

f = open(rootPath + "\\yolov5\\result.txt", "w")

check_list = []
for file in result_file:
    result = open(file, 'r')
    for line in result:
        f.write(line)
f.close()
