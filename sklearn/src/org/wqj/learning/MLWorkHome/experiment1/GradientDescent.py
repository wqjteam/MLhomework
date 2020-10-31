# coding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import genfromtxt
dataPath = r"./Input/data1.csv"
dataSet = pd.read_csv(dataPath,header=None)
print(dataSet)
price = []
rooms = []
area = []
for data in range(0,len(dataSet)):
    area.append(dataSet[0][data])
    rooms.append(dataSet[1][data])
    price.append(dataSet[2][data])

def gradientDescent(rooms, price, area):
    theta = []
    return theta