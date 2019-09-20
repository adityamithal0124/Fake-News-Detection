import DataExtraction
import FeatureExtraction
from sklearn.model_selection import ShuffleSplit
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

naive = Pipeline([('NB',FeatureExtraction.count),('nb',MultinomialNB())])
naive.fit(DataExtraction.train2['Statement'],DataExtraction.train2['Label'])
pred = naive.predict(DataExtraction.test2['Statement'])
np.mean(pred == DataExtraction.test2['Label'])

Logistic = Pipeline([('LR',FeatureExtraction.count), ('lr',LogisticRegression())])
Logistic.fit(DataExtraction.train2['Statement'],DataExtraction.train2['Label'])
predLR = Logistic.predict(DataExtraction.test2['Statement'])
np.mean(predLR == DataExtraction.test2['Label'])

def matrix(classifier):
    k_fold = KFold(n_splits = 5)
    final = []
    conf = np.array([[0,0],[0,0]])
    for tr,te in k_fold.split(DataExtraction.train2):
        tr_text = DataExtraction.train2.iloc[tr]['Statement']
        tr_y = DataExtraction.train2.iloc[tr]['Label']
        
        te_text = DataExtraction.train2.iloc[te]['Statement']
        te_y = DataExtraction.train2.iloc[te]['Label']
        
        classifier.fit(tr_text,tr_y)
        pred = classifier.predict(te_text)
        
        conf += confusion_matrix(te_y,pred)
        result = f1_score(te_y,pred,pos_label = 'TRUE')
        final.append(result)
    
    return (print('Classified: ',len(DataExtraction.train2)),print('Score: ',sum(final)/len(final)),print('Result Length: ',len(final)),print('Confusion Matrix: '),print(conf))

matrix(naive)

matrix(Logistic)

model_file = 'final_model.sav'
pickle.dump(naive,open(model_file,'wb'))

