
from tkinter import Tk                                                         #import tkinter
from tkinter.filedialog import askopenfilename
import pandas as pd                                                            #import pandas
#import numpy as np                                                            #import numpy
#import matplotlib.pyplot as plt                                               #import matpotlib
from matplotlib.pylab import rcParams 
from sklearn.preprocessing import MinMaxScaler                   
def Out():                                                                     #function returns file path
    Tk().withdraw()                                                            #we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename()                                               #shows open dialogue box and returns file path
    return filename
rcParams['figure.figsize'] = 20,10                                             #Setting Figure Size           
scaler = MinMaxScaler(feature_range=(0, 1))                                    #For Normalizing Data
df = pd.read_csv(Out())                                                        #read the file
print(df)                                                                      #print csv file