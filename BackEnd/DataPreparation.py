import pandas as pd
import numpy as np
import pickle

import csv

from nltk.tokenize import RegexpTokenizer
#from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim

from nltk.corpus import stopwords
tokenizer = RegexpTokenizer(r'\w+')

from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

from collections import Counter

import numpy as np
import lda
import lda.datasets
import numpy as np
import textmining
import time

###############################
###
### This File takes the Headphones dataFrame and outputs 3 files
###
### editedT.pkl a file that containes all of the edited reviews 
### vocab.csv which contains the vocab list
### lda_X.npy which is the corpus the LDA will be trained on
###
#########

def load_relevant_files():
    '''loads dataframe, stopwords, sentiment words from files'''
    
    dfHeadphones=pd.DataFrame.from_csv('dfHeadphonesRR.csv')
    
    with open('EnglishStops.csv', 'rb') as f:
    	reader = csv.reader(f)
    	your_list = list(reader)
    englishstops=your_list[0]
    
    with open ("positive-words.txt", "r") as myfile:
    	datap=myfile.readlines()
    with open ("negative-words.txt", "r") as myfile:
    	datan=myfile.readlines()


    i=0
    pos=[]
    for line in datap : 
        i=i+1
        if i<36 : continue
        pos.append(line.strip())
    i=0
    neg=[]
    for line in datan : 
        i=i+1
        if i<36 : continue
        neg.append(line.strip())
        
    return dfHeadphones, englishstops, pos, neg


def prepare_data_1(df=[],en_stops=[],pos=[],neg=[]):
    '''gets data into preparation mode for LDA training
    this part is mainly removing unwanted words and injecting sentiment codewords'''
    
    doc_set=[]
    for i in range(0,len(df['Review'])) :
        doc_set.append(df['Review'].values[i])

    texts = []

    # loop through document list
    j=-1
    for i in doc_set:
        j=j+1
    # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)

        stopped_tokens = [i for i in tokens if not i in en_stops]
        
        texts.append(stopped_tokens)
    
        if (j%5000==0) : print (float(j)/len(df['Review']))


    print "Stops removed"
    
    fulllist = []
    for i in range(0,len(texts)):
        for j in texts[i] :
            fulllist.append(j)
    most_common_words= [word for word, word_count in Counter(fulllist).most_common(10000)]


    print "10000 Most Common words Found"
    
    editedT=[]
    i5=0
    maxl=len(texts)

    start_time = time.time()

    print "Injecting Sentiment Codewords"

    for iline in range(0,maxl) :
        line=texts[iline]
        i5=i5+1
        if i5>50000 : break
        if (i5%1000==0) : print(float(i5)/50000),time.time()-start_time
        lines = [i for i in line if i in most_common_words]
        newline=[]
        for aword in lines :
            if aword in pos : 
                newline.append(aword)
                newline.append('GOODREVIEW')
            elif aword in neg :
                newline.append(aword)
                newline.append('BADREVIEW')
            else : newline.append(aword)
        A = ' '.join(word for word in newline)
        editedT.append(A)
    timef=time.time()
    print("--- %s seconds ---" % (timef - start_time))
    
    print "Done: Ready for Step 3"
    
    with open('check_test1.csv','wb') as file:
        for line in editedT:
           file.write(line)
           file.write('\n') # save progress along the way to deal with crashes later on
    
    return editedT

def prepare_data_2(clean_reviews=[]):
    '''prepares reviews by creating the LDA corpus'''
    
    tdm = textmining.TermDocumentMatrix()
    for doc1 in clean_reviews :
        tdm.add_doc(doc1)
    temp = list(tdm.rows(cutoff=2))
    vocab = tuple(temp[0])
    X = np.array(temp[1:])
    
    return X,vocab  







