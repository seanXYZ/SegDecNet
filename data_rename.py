import glob
import os
from os.path import exists, isdir, split

dataRoot = "/home/sean/Data/KolektorSDD_sean/temp"


for x in glob.glob(dataRoot + "/*"):
    if isdir(x):
        for y in glob.glob(x + "/*"):
            strs = y.split('/')
            newName = os.path.join(x, strs[-2] + '_' + strs[-1] )
            os.rename(y, newName)
