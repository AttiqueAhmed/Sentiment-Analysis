#importing necessary libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from keras.callbacks import EarlyStopping
import warnings
from nltk.stem.wordnet import WordNetLemmatizer
import re, string, nltk
import emoji, bz2
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.embeddings import Embedding
from sklearn.metrics import roc_auc_score


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


lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub('[^a-zA-Z0-9]',' ',text)
    text= re.sub(emoji.get_emoji_regexp(),"",text)
    text = [lemmatizer.lemmatize(word) for word in text.split() if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text

data2["clean_review"] = data2["review"].apply(preprocess_text)

#replacing the label values
data2.label.replace({1:0, 2:1}, inplace = True)
data2.head()

#selecting only labels and clean reviews
data_review = data2[['label', 'clean_review']]
data_review

#splitting data for test and train

X_train, X_test, y_train, y_test = train_test_split(np.array(data_review["clean_review"]),np.array(data_review["label"]), test_size=0.30,random_state= 5)
print(X_train.shape)
print(X_test.shape)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

#padding to keep the number of words 
X_train_seq_padded = pad_sequences(X_train_seq, maxlen=64)
X_test_seq_padded = pad_sequences(X_test_seq, maxlen=64)



tokenizer = Tokenizer()

tokenizer.fit_on_texts(X_train)

# using tokenizer to transform reviews into training and testing set
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_seq_padded = pad_sequences(X_train_seq, maxlen=64)
X_test_seq_padded = pad_sequences(X_test_seq, maxlen=64)



BATCH_SIZE = 32

model = Sequential()
model.add(Embedding(len(tokenizer.index_word)+1, 64))
model.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(128, activation="relu"))
model.add(Dense(1,activation="sigmoid"))

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics=["accuracy"])
model.summary()


early_stop = EarlyStopping(monitor="val_loss", patience=5, verbose=True)

history = model.fit(X_train_seq_padded, y_train,batch_size=BATCH_SIZE,epochs=8,
                    validation_data=(X_test_seq_padded, y_test),callbacks=[early_stop])

# Calculating the accuracies of model for test and train predictions

pred_train = model.predict(X_train_seq_padded)
pred_test = model.predict(X_test_seq_padded)
print('LSTM Train ROC score: ' + str(roc_auc_score(y_train, pred_train)))
print('LSTM Test ROC Score: ' + str(roc_auc_score(y_test, pred_test)))

# evaluating the model 
model.evaluate(X_test_seq_padded, y_test)

# plotting accuracy
acc = history.history["accuracy"]
loss = history.history["loss"]

val_acc = history.history["val_accuracy"]
val_loss = history.history["val_loss"]

plt.figure(figsize=(9,6))
plt.plot(acc,label="Training Accuracy")
plt.plot(val_acc,label="Validation Accuracy")
plt.legend()
plt.xlabel("No. of Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")

#plotting loss
plt.figure(figsize=(9,6))
plt.plot(loss,label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.legend()
plt.xlabel("No. of Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")

"""from the above graphs, we can see the accuracy as well as the loss for both validation and traing data. 
From the first graph, it is visible that the validation accuracy over the epochs has decreased as compared 
to training accuracy. Same as validation loss has increased comapared to validation loss."""

#testing model for a manual input
rvw = ['i hate eating']
#vectorizing the review by the pre-fitted tokenizer instance
rvw = tokenizer.texts_to_sequences(rvw)
#padding the review to have exactly the same shape as `embedding_2` input
rvw = pad_sequences(rvw, maxlen=28)
sentiment = model.predict(rvw,batch_size= 1, verbose = 2)
print(sentiment)
print(np.rint(sentiment))
print(np.argmax(sentiment))
if(np.rint(sentiment) == 0):
    print("negative")
elif (np.rint(sentiment) == 1):
    print("positive")

