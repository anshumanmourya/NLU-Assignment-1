
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import brown
from nltk.corpus import gutenberg
from sklearn.model_selection import train_test_split
import itertools
import string
import math


# In[2]:


START = '<s>'
STOP = '</s>'
UNK = '<unk>'
l1 = 0.6
l2 = 0.2
l3 = 0.2
punc = list(string.punctuation)


# In[3]:


def preprocess_Train_data(data):
    for line in data :
        line.insert(0,START)
        line.insert(0,START)
        line.append(STOP)
    words = list(itertools.chain.from_iterable(data))
    data_words = [x.lower() for x in words]
    data_words.append(UNK)
    freq = nltk.FreqDist(data_words)
    unknown = list()
    for word in freq.keys():
        if freq[word] == 1:
            unknown.append(word)
    unknown = set(unknown)
    vocab = list()
    for w in data_words :
        if w in unknown :
            vocab.append(UNK)
        else :
            vocab.append(w)
    data_words = vocab
    return data , words, data_words ,unknown


# In[4]:


def create_bigram_table( data_words ):
    cfreq_2gram = nltk.ConditionalFreqDist(nltk.bigrams(data_words))
    cprob_2gram = nltk.ConditionalProbDist(cfreq_2gram, nltk.MLEProbDist)
    return cfreq_2gram,cprob_2gram


# In[5]:


def create_trigram_table( data_words ):
    trigrams = nltk.trigrams(data_words)
    sen = list()
    sen = [((a,b),c) for (a,b,c) in trigrams]
    cfreq_3gram = nltk.ConditionalFreqDist(sen)
    cprob_3gram = nltk.ConditionalProbDist(cfreq_3gram, nltk.MLEProbDist)
    return cfreq_3gram,cprob_3gram    


# In[6]:


def unigram_prob ( word, freq_1gram,len_):
    return freq_1gram[ word] / float(len_)


# In[7]:


def backoff ( w1 , w2 , w3 , cprob_3gram , cprob_2gram ,freq_1gram,len_):
        if cprob_3gram[(w1,w2)].prob(w3) > 0:
            #print 3 
            return l1*cprob_3gram[(w1,w2)].prob(w3)
        else:
            if cprob_2gram[w2].prob(w3) > 0:
                #print 2
                return 0.4*l2*cprob_2gram[w2].prob(w3)
            else :
                #print 1
                return 0.4*l3*unigram_prob(w3,freq_1gram,len_)


# In[8]:


def interpolate( w1 , w2 , w3 , cprob_3gram , cprob_2gram ,data_words):
    x = cprob_3gram[(w1,w2)].prob(w3)
    y = cprob_2gram[w2].prob(w3)
    z = unigram_prob(w3 ,data_words)
    return l1*x + l2*y + l3*z


# In[9]:


def preprocess_test(Test,vocab_set):
    test = []
    for line in Test :
        t = []
        for word in line :
            if word not in vocab_set :
                t.append(UNK)
            else :
                t.append(word)
        #t = [i.lower() for i in line if i not in punc]
        t.append(STOP)
        t.insert(0,START)
        t.insert(0,START)
        test.append(t)
    return test


# In[10]:


def evaluate_perplexity(test,cprob_3gram,cprob_2gram,freq_1gram,len_):
    perp = 0
    n = 0
    for line in test :
        for i in range(2,len(line)):
            val = backoff(line[i-2],line[i-1],line[i],cprob_3gram,cprob_2gram,freq_1gram,len_)
            #val = interpolate(line[i-2],line[i-1],line[i],cprob_3gram,cprob_2gram,data_words)
            perp += math.log(val,2)
        n += len(line)
    perp = (-1) * (perp / float(n))
    return  2**perp


# In[11]:


def create_model(Train , Test):
    Train, words, Train_words, unknown = preprocess_Train_data ( Train)
    print "step1"
    vocab_set = set( Train_words)
    print "step2"
    freq_1gram = nltk.FreqDist(Train_words)
    len_ = len(Train_words)
    print "step3"
    cfreq_2gram, cprob_2gram = create_bigram_table ( Train_words)
    print "step3"
    cfreq_3gram, cprob_3gram = create_trigram_table ( Train_words)
    print "step4"
    test = preprocess_test ( Test,vocab_set)
    print "step5"
    return cprob_3gram,cprob_2gram,freq_1gram,len_,test 


# In[12]:


def try_dataset(dataset):
    if dataset == "brown":
        brown_sent = brown.sents()
        D1_Train,D1_Test = train_test_split(brown_sent,train_size = 0.8 , random_state = 7)    
        return create_model(D1_Train,D1_Test)

    else :
        if dataset == "gutenberg":
            gut_sent = gutenberg.sents()
            D2_Train,D2_Test = train_test_split(gut_sent,train_size = 0.8 , random_state = 7)
            return create_model(D2_Train,D2_Test)
        else:
            if dataset == "testD1":
                brown_sent = brown.sents()
                gut_sent = gutenberg.sents()
                D3_Train,D3_Test = train_test_split(brown_sent,train_size = 0.8 , random_state = 7)
                D3_Train = gut_sent + D3_Train
                return create_model(D3_Train,D3_Test)
            else :
                brown_sent = brown.sents()
                gut_sent = gutenberg.sents()
                D4_Train,D4_Test = train_test_split(gut_sent,train_size = 0.8 , random_state = 7)
                D4_Train = brown_sent + D4_Train
                return create_model(D4_Train,D4_Test)


# In[13]:


print("Perplexity on Brown dataset")
bcprob_3gram,bcprob_2gram,bfreq_1gram,blen_,btest = try_dataset("brown")


# In[14]:


perplexity1 = evaluate_perplexity (btest ,bcprob_3gram, bcprob_2gram, bfreq_1gram,blen_)
print perplexity1


# In[15]:


print("Perplexity on Gutenberg dataset")
gcprob_3gram,gcprob_2gram,gfreq_1gram,glen_,gtest = try_dataset("gutenberg")


# In[16]:


perplexity2 = evaluate_perplexity (gtest ,gcprob_3gram, gcprob_2gram, gfreq_1gram,glen_)
print perplexity2


# In[17]:


print("Perplexity on Brown+Gutenberg test on Brown dataset")
mcprob_3gram,mcprob_2gram,mfreq_1gram,mlen_,mtest = try_dataset("testD1")


# In[18]:


perplexity3 = evaluate_perplexity (mtest ,mcprob_3gram, mcprob_2gram, mfreq_1gram,mlen_)
print perplexity3


# In[19]:


print("Perplexity on Brown+Gutenberg test on Gutenberg dataset")
mcprob_3gram2,mcprob_2gram2,mfreq_1gram2,mlen_2,mtest2 = try_dataset("testD2")


# In[20]:


perplexity4 = evaluate_perplexity (mtest2 ,mcprob_3gram2, mcprob_2gram2, mfreq_1gram2,mlen_2)
print perplexity4

