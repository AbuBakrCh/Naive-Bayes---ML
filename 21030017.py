#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re as re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

def preprocess(stopWords, dirPath):    
    processedReviews = []
    for review in Path(dirPath).iterdir():
        review = review.read_text().lower()
        review = cleanStopWords(review)
        processedReviews += [' '.join(re.findall(r'\b([a-z]+)\b', re.sub('[(),?!.;:\n]|<br /><br />','', review)))]
    return processedReviews
        
def cleanStopWords(review):
    for stopword in stopWords.iloc[:,0]:
        review = re.sub(rf'\b{stopword}\b', '', review)
    return review

def generateVocabulary(trainReviews):
    global vocabulary
    for review in trainReviews:
        vocabulary.update(set(review.split()))


# In[2]:


vocabulary = set()

stopWordsDir = '/home/abu-bakr/Documents/ML@LUMS/Assignment 4/Dataset/stop_words.txt'
stopWords = pd.read_csv(stopWordsDir, header=None)

trainPosDir = '/home/abu-bakr/Documents/ML@LUMS/Assignment 4/Dataset/train/pos'
trainPositiveReviews = preprocess(stopWords, trainPosDir)
generateVocabulary(trainPositiveReviews)
trainPositiveTokens = [word for review in trainPositiveReviews for word in review.split()]

trainNegDir = '/home/abu-bakr/Documents/ML@LUMS/Assignment 4/Dataset/train/neg'
trainNegativeReviews = preprocess(stopWords, trainNegDir)
generateVocabulary(trainNegativeReviews)
trainNegativeTokens = [word for review in trainNegativeReviews for word in review.split()]


# In[ ]:





# In[3]:


def calcVocabularyLikelihood(wordCountDict):
    global vocabulary
    likelihood = dict()
    laplaceDenominator = sum(wordCountDict.values()) + len(vocabulary)
    for word in vocabulary:
        likelihood[word] = np.log((wordCountDict[word] + 1)/laplaceDenominator)
    return likelihood


# In[4]:


numDocuments = 25000
numPositive = 12500
numNegative = 12500

logPrior = {'Positive': np.log(numPositive/numDocuments), 'Negative': np.log(numNegative/numDocuments)}


posWordCountDict = Counter(trainPositiveTokens)
negWordCountDict = Counter(trainNegativeTokens)



logLikelihoodPosClass = calcVocabularyLikelihood(posWordCountDict)
logLikelihoodNegClass = calcVocabularyLikelihood(negWordCountDict)


# In[ ]:





# In[5]:


def testClassification(testReviews, logLikelihoodPosClass, logLikelihoodNegClass, logPrior):
    global vocabulary
    predictions = {1: 0, -1: 0}
    for review in testReviews:
        posteriorPos = logPrior['Positive']
        posteriorNeg = logPrior['Negative']
        for word in review.split():
            if word in vocabulary:
                posteriorPos += logLikelihoodPosClass[word]
                posteriorNeg += logLikelihoodNegClass[word]
        if(posteriorPos >= posteriorNeg):
            predictions[1] = predictions[1] + 1
        else:
            predictions[-1] = predictions[-1] + 1
    return predictions


def calculateConfusionMatrix(predictedLabelsForPositiveClass, predictedLabelsForNegativeClass):
    confusionMatrix = np.empty(shape=(2,2))
    #for below dictionary for positive instances, key:1 contains the count of predicted positives
    confusionMatrix[0][0] = predictedLabelsForPositiveClass[1]
    #for below dictionary for negative instances, key:1 contains the count of predicted positives
    confusionMatrix[0][1] = predictedLabelsForNegativeClass[1]
    #for below dictionary for positive instances, key:-1 contains the count of predicted negatives
    confusionMatrix[1][0] = predictedLabelsForPositiveClass[-1]
    #for below dictionary for negative instances, key:-1 contains the count of predicted negatives
    confusionMatrix[1][1] = predictedLabelsForNegativeClass[-1]
    
    confusionMatrix[np.isnan(confusionMatrix)] = 0
    return confusionMatrix


# In[6]:


testPosDir = '/home/abu-bakr/Documents/ML@LUMS/Assignment 4/Dataset/test/pos'
testPositiveReviews = preprocess(stopWords, testPosDir)

testNegDir = '/home/abu-bakr/Documents/ML@LUMS/Assignment 4/Dataset/test/neg'
testNegativeReviews = preprocess(stopWords, testNegDir)


# In[9]:


predictedLabelsForPositiveClass = testClassification(testPositiveReviews, logLikelihoodPosClass, logLikelihoodNegClass, logPrior)
predictedLabelsForNegativeClass = testClassification(testNegativeReviews, logLikelihoodPosClass, logLikelihoodNegClass, logPrior)

confusionMatrix = calculateConfusionMatrix(predictedLabelsForPositiveClass, predictedLabelsForNegativeClass)
print("Confusion Matrix: \n", confusionMatrix)
accuracy = (confusionMatrix[0][0] + confusionMatrix[1][1]) / np.sum(confusionMatrix)
print("Accuracy:", accuracy)


# In[8]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix

vectorizer = CountVectorizer()

trainAllReviews = trainPositiveReviews + trainNegativeReviews
X_train = vectorizer.fit_transform(trainAllReviews)

trainPositiveLabels = [1] * len(trainPositiveReviews)
trainNegativeLabels = [-1] * len(trainNegativeReviews)
y_train = trainPositiveLabels + trainNegativeLabels

testAllReviews = testPositiveReviews + testNegativeReviews
X_test = vectorizer.transform(testAllReviews)

testPositiveLabels = [1] * len(testPositiveReviews)
testNegativeLabels = [-1] * len(testNegativeReviews)
y_test = testPositiveLabels + testNegativeLabels

mnb = MultinomialNB()
mnb.fit(X_train,y_train)
mnb_predicted = mnb.predict(X_test)

#Please be lenient while penalizing. Complete assignment was submitted just an hour late. First draft
#was on time. It had only slightly incomplete part 2.

print("Accuracy:",metrics.accuracy_score(y_test, mnb_predicted))
print("Confusion Matrix: \n", confusion_matrix(y_test, mnb_predicted))

