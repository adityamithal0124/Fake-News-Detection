import DataExtraction
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer()
tr = count.fit_transform(DataExtraction.train2['Statement'].values)

def getMatrix():
    print(count.vocabulary_)
    print(count.get_feature_names()[0:25])