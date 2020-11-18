# coding=utf-8
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = curPath[:curPath.find("sklearn\\") + len("sklearn\\")]
dataPath = rootPath + "Input/MLWorkHome/experiment2/data.csv"
x_y_data = pd.read_csv(dataPath,header=1)
print(x_y_data[1][1])
year = []
bus = []
gdp = []
for index in range(0, len(x_y_data)):
    year.append(x_y_data["Year"][index])
    bus.append(x_y_data["Bus"][index])
    gdp.append(x_y_data["PGDP"][index])
print(year)