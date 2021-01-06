

import glob
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = curPath[:curPath.find("ultimatemlwork\\") + len("ultimatemlwork\\")]
label_path = rootPath+"\\yolov5\\runs\\detect\\exp18\\"
jpg_path = rootPath+"\\yolov5\\runs\\detect\\exp18\\"
result_file = glob.glob(label_path + '*.txt')
#jpg_file = glob.glob(jpg_path + '*.jpg')

f = open(rootPath+"\\yolov5\\result.txt","w")



check_list = []
for file in result_file:
    result = open(file,'r')
    for line in result:
        f.write(line)
f.close()


