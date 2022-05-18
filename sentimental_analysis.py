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

#we will also check for any dupilcates 
data.duplicated().sum()

#sentiment count
data.label.value_counts()

#showing data to clean
data2 = data.sample(frac = 0.50, random_state= 0)
data2 = data2.drop_duplicates(ignore_index = True)

data2.label.value_counts()
#plot the label count
sns.countplot(data2.label)
plt.title("Count of labels")
#function to clean the reviews
def clean_text(df, field):
    df[field] = df[field].str.replace(r"@"," at ")
    df[field] = df[field].str.replace("#[^a-zA-Z0-9_]+"," ")
    df[field] = df[field].str.replace(r"[^a-zA-Z(),\"'\n_]"," ")
    df[field] = df[field].str.replace(r"http\S+","")
    df[field] = df[field].str.lower()
    return df

clean_text(data2,"review")

from nltk.stem.wordnet import WordNetLemmatizer
import re, string, nltk
import emoji, bz2
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub('[^a-zA-Z0-9]',' ',text)
    text= re.sub(emoji.get_emoji_regexp(),"",text)
    text = [lemmatizer.lemmatize(word) for word in text.split() if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text
