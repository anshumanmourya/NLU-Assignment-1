
# coding: utf-8

# In[384]:


import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import gutenberg
from nltk.corpus import brown
from sklearn.model_selection import train_test_split


# In[385]:


START = '<s>'
STOP = '</s>'
UNK = '<unk>'
gut_sent = gutenberg.sents()
brown_sent = brown.sents()
d = gut_sent+brown_sent
#d = brown_sent


# In[386]:


data = []
for line in d :
    line.insert(0,START)
    line.insert(0,START)
    line.append(STOP)
    data.append(line)


# In[387]:


import itertools
words = list(itertools.chain.from_iterable(data))

import string 
punc = list(string.punctuation)
punc.remove('.')
punc.remove(',')
#print punc


# In[388]:


#data_words = [i.strip("".join(punc)) for i in words if i not in punc]
data_words = [x for x in words if x not in punc]
#print data_words[0]
data_words.append(UNK)

freq_gut = nltk.FreqDist(data_words)

unknown = list()
for word in freq_gut.keys():
    if freq_gut[word] == 1:
        unknown.append(word)
unknown = set(unknown)

vocab = list()
for w in data_words :
    if w in unknown :
        vocab.append(UNK)
    else :
        vocab.append(w)
data_words = vocab

data_vocab_set = set(data_words)

trigrams = nltk.trigrams(data_words)
sen = list()
sen = [((a,b),c) for (a,b,c) in trigrams] 

cfreq_gut_3gram = nltk.ConditionalFreqDist(sen)

cprob_gut_3gram = nltk.ConditionalProbDist(cfreq_gut_3gram, nltk.MLEProbDist)


# In[396]:


i=0
x = START
y = START
while i<10 :
    s =  cprob_gut_3gram[(y,x)].generate()
    if s not in [START,STOP,UNK," "]+punc:
        #print "not to print = "+s
        print s, #+ end=' ')
        y = x
        x = s
        #if s not in punc:
        i+=1    
    if s == STOP or s == UNK:
        #print ".",# + end=' ')
        y = START
        x = START
    if (i==10):
        print "."

