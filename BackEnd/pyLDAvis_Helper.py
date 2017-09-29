from nltk.tokenize import RegexpTokenizer
import numpy as np

tokenizer = RegexpTokenizer(r'\w+')



def Create_lengths(editedT=[]):
    '''calculates the length of each corpus'''
    tokensT=[]
    for i in editedT:
        tokens2=[]
    # clean and tokenize document string
        raw = i.lower()
        tokens2 = tokenizer.tokenize(raw)
        tokensT.append(tokens2)
    lengths=[]
    for tok in tokensT:
        lengths.append(len(tok))

    print 'lengths complete'
    return lengths


def Create_vocabclicks(X=[]):
	''' calculates the amount of times each word appears in the corpus'''
	vocabclicks=np.sum(X, axis=0)
	print 'vocabs complete'
	return vocabclicks