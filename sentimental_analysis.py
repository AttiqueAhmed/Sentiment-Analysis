#importing necessary libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

#to avoid any warnings
import warnings
warnings.filterwarnings('ignore')

#importing the data 
data = pd.read_csv('test.csv', header = None)
data.head(10)

#Setting the headers 
headers = ['label', 'title', 'review']
data = pd.read_csv('test.csv', names = headers)
data.head()

#we will check the shape of the data 
print(f"The data has {data.shape[0]} rows and {data.shape[1]} columns.]")

#checking for any null values 
data.isna().sum()

#dropping the null entries
data = data.dropna()
data

"""#we will also check for any dupilcates 
data.duplicated().sum()

#sentiment count
data.label.value_counts()

#showing data to clean
data2 = data.sample(frac = 0.50, random_state= 0)
data2 = data2.drop_duplicates(ignore_index = True)
data2

data2.label.value_counts()"""
