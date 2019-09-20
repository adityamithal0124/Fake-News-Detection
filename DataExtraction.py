import numpy as np
import pandas as pd
import seaborn as sb
import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer

test2_file = 'test2.csv'
train2_file = 'train2.csv'
val2_file = 'val2.csv'

test2 = pd.read_csv(test2_file)
train2 = pd.read_csv(train2_file)
val2 = pd.read_csv(val2_file)

def graph(file):
    return sb.countplot(x='Label',data = file, palette = 'hls')

def stemming_token(stemmer,tokens):
    stemmed = []
    for t in tokens:
        stemmed.append(stemmer.stem(t))
    return stemmed

eng = SnowballStemmer('english')
stopwords = set(nltk.corpus.stopwords.words('english'))

def processing(data,exclude_stopword=True,stem = True):
    for i in data:
        tokens.append(i.lower())
    tokens_stemmed = tokens
    tokens_stemmed = stemming_token(eng,tokens)
    tokens_stemmed = [j for j in tokens_stemmed if i not in stopwords]

def unigram(words):
    assert type(words) == list
    return words

def bigram(words):
    assert type(words) == list
    skip = 0
    s = " "
    length = len(words)
    if(len >1):
        lst = []
        for i in range(length-1):
            for j in range(1,skip+2):
                if(i+j < length):
                    lst.append(s.join([words[i],words[i+j]]))
    else:
        lst = unigram(words)
    return lst

def tokenizing(text):
    return text.split()


def port_token(text):
    return [porter.stem(word) for word in text.split()]

for i in range(len(test2)):
    if(test2.iloc[i,2] == "TRUE" or test2.iloc[i,2] == 'half-true' or test2.iloc[i,2] == 'mostly-true'):
        test2.iloc[i,2] = 'TRUE'
    else:
        test2.iloc[i,2] = 'FALSE'


for i in range(len(train2)):
    if(train2.iloc[i,2] == "TRUE" or train2.iloc[i,2] == 'half-true' or train2.iloc[i,2] == 'mostly-true'):
        train2.iloc[i,2] = 'TRUE'
    else:
        train2.iloc[i,2] = 'FALSE'

for i in range(len(val2)):
    if(val2.iloc[i,2] == "TRUE" or val2.iloc[i,2] == 'half-true' or val2.iloc[i,2] == 'mostly-true'):
        val2.iloc[i,2] = 'TRUE'
    else:
        val2.iloc[i,2] = 'FALSE'








