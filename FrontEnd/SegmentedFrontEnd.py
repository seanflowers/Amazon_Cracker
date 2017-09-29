from bs4 import BeautifulSoup
import re
import urllib2
import csv
import time
import numpy as np
import lda
import lda.datasets
import numpy as np
import textmining
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim

from scipy.spatial import distance

from nltk.corpus import stopwords
tokenizer = RegexpTokenizer(r'\w+')

import pickle


import sys

#import matplotlib.pyplot as plt

fr=file('regressor_sep20.pkl','rb')
clf=pickle.load(fr)

def Check_variable_response(V_h=[]):
    '''Check the response of each variable on the classifier output'''
    response=[]
    V_h_Out=clf.predict(np.array(V_h).reshape(1,-1))
    for v in range(0,len(V_h)):
        Temp=list(V_h)
        Temp[v]=V_h[v]*1.02
        TempHighOut=clf.predict(np.array(Temp).reshape(1,-1))
        Temp[v]=V_h[v]*0.08
        TempLowOut=clf.predict(np.array(Temp).reshape(1,-1))
        
        AvgResp=(abs(TempHighOut-V_h_Out)+abs(TempLowOut-V_h_Out))/2.
        WeightedResponse=AvgResp/V_h_Out
        
        response.append(WeightedResponse[0])
    return response


def find_best_shoot(darray=[],GoalStars=5.0,number_of_shoots=100):
    ''' Create Random off Shoots from product in topic space and calculate the smallest'''
    V_h = darray
    Shoots=[]
    # Create a random direction to move in
    for s in range(0,number_of_shoots):
     
        C_rand=(np.random.rand(1,len(V_h))*0.2)[0]
        C_s=np.multiply(C_rand,V_h)
        
        for n in range(1,6):
            V_s_n=[]
            for i in range(0,len(V_h)):
                if i<10 : V_s_n.append(V_h[i]+n*C_s[i]) #Improve Good Attributes
                else : V_s_n.append(V_h[i]-n*C_s[i]) # Reduce Bad Attributes
            #print n, V_s_n
            R_s_n=clf.predict(np.array(V_s_n).reshape(1,-1))
            if(abs(R_s_n-GoalStars)<0.2) : 
                Shoots.append(V_s_n)
                break 
    Improvement_V=np.array(Shoots[distance.cdist([V_h], Shoots).argmin()])-np.array(V_h)
    return np.divide(Improvement_V,V_h)*100

#def PLOT_PRODUCT_TOPIC(data,typ):
    '''Create product breakdown and improvement plots'''
 #   labels = ['Good Reviews','Good Cable/Cord','Good Handsfree/Wireless', 'Good Levels','Good Comfort/Fit',
  #            'Good Durability','Good Sound','Good Case','Good Mic','Good Value','Bad Value','Bad Durability',
   #           'Bad Cable/Cord','Bad Reviews','Bad Customer Service','Bad Handsfree/Wireless','Bad Comfort/Fit',
    #          'Bad Sound']

#    coloring=['g','g','g','g','g','g','g','g','g','g','g','r','r','r','r','r','r','r','r']
#    xlocations = np.array(range(len(data)))+0.5
#    width = 0.5
#    plt.bar(xlocations, data, width=width,color=coloring)
#    plt.yticks(range(0, 1))
#    plt.xticks(xlocations+ width/2, labels,fontsize=20,rotation='vertical')
#    plt.xlim(0, xlocations[-1]+width*2)
#    if typ == 1 : 
 #   	plt.ylim(0,np.array(data).max()+.05)
 #   	plt.title("Product Breakdown",fontsize=20)
#    	plt.yticks(np.arange(0,1,0.1))
#    if typ == 2 :
#    	plt.ylim(np.array(data).min()-.02,np.array(data).max()+.02)
#    	plt.title("Improvement Strategy",fontsize=20)
#    	plt.yticks(np.arange(-1,1,0.1))
#    plt.axhline(0, color='black')
#    plt.gca().get_xaxis().tick_bottom()
#    plt.gca().get_yaxis().tick_left()
#    fig = plt.gcf()
#    fig.set_size_inches(20, 8.5)
#    plt.tight_layout()
#    if type==1:plt.savefig('flaskexample/static/img/ProductBreakdown.png')
#    if type==2:plt.savefig('flaskexample/static/img/ProductImprovement.png')

##tester B013JDQFDU
##test 'https://www.amazon.com/Samsung-Wired-Headset-Galaxy-Packaging/dp/B01CO60NAO/ref=cm_cr_arp_d_product_top?ie=UTF8'

######################################
#    Step 1: Collecting Reviews
#######################################

def Collect_Reviews(url_input=''):
    """ Scrape amazon.com for reviews for given input product"""
    print 'a change has been made'

    input_url=url_input
    #input_url=sys.argv[1]

    Product_Name=[]
    Product_Code=[]
    f=0
    for j in range(0,len(input_url)) :
        if j<23 : continue
        if f==0 : Product_Name.append(input_url[j])
        if f==2 : Product_Code.append(input_url[j+1])
        if ((input_url[j+2]=='/') & (f==2)): break
        if ((input_url[j+1]=='/') & (f==1)): f=2
        if ((input_url[j+1]=='/') & (f==0)): f=1


    PName=''.join(Product_Name)
    PCode=''.join(Product_Code)

    url = 'https://www.amazon.com/product-reviews/' + PCode + '/' ##

    #url = 'https://www.amazon.com/product-reviews/' + sys.argv[1] + '/' ##
    print url
    print 'Collecting Reviews...'

    
    hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11'}
    req = urllib2.Request(url, headers=hdr)
    soup_Baker = BeautifulSoup(urllib2.urlopen(req), "html5lib")
    Reviews=[]
#Find Product Info
    RobotBlocks=0
    for link in soup_Baker.find_all('title',{'dir':"ltr"}) :
            if link.text =='Robot Check' : RobotBlocks=RobotBlocks+1


    STAR=0
    star=['3']
    for link in soup_Baker.find_all( 'span' ,{"class" : "arp-rating-out-of-text"}):
        star=[]
        star.append(link.text[0])
        star.append(link.text[1])
        star.append(link.text[2])
    STAR=float(''.join(star))
    price=[0]
    for link in soup_Baker.find_all( 'span' ,{"class" : "a-color-price arp-price"}):
        price=link.text
    PRICE=float(price[1:])
#Get Reviews From First Page
    for link in soup_Baker.find_all( 'div' ,{"class" : "a-section review"}):
       for link2 in link.find_all('span',{"data-hook":"review-body"}):
            Reviews.append(link2.text)
#Get Page Info
    il=0
    for link in soup_Baker.find_all('li',{"class" : "page-button"}):
        il=il+1
        for a in link.find_all('a', href=True):
            if il==2 : newURL='https://www.amazon.com'+a['href']
            if il==5 : pagesN=int(link.text)
    
    for link in soup_Baker.find_all('title',{'dir':"ltr"}) :
        if link.text =='Robot Check' : RobotBlocks=RobotBlocks+1

            
#LOOP TO STEAL!
    for jl in range(2,pagesN-1):
        print 'page',jl,'of',pagesN-1,' : Reviews = ',len(Reviews)
        print 'URL===',newURL
        
        
        req = urllib2.Request(newURL, headers=hdr)
        con = urllib2.urlopen( req )


        soup_Baker = BeautifulSoup(con, "html5lib")
        for link in soup_Baker.find_all( 'div' ,{"class" : "a-row review-data"},limit=5):
            Reviews.append(link.text)
        for link in soup_Baker.find_all('title',{'dir':"ltr"}) :
            if link.text =='Robot Check' : RobotBlocks=RobotBlocks+1
        k=jl+1
        if jl<10 : newURL=newURL[0:-22]+str(k)+newURL[-21:-1]+str(k)
        if jl>10 : newURL=newURL[0:-24]+str(k)+newURL[-22:-2]+str(k)
        
        if jl==99 : break
        print 'Robots thwarted:',RobotBlocks,'of', pagesN-1

    print 'Done! ',len(Reviews),' collected!'
    return Reviews, STAR, PName
######################################
#    Step 2: Sentimentizing Reviews
#######################################

def Sentimentize_Reviews(Reviews=[]):
    """ Analyze Reviews for Sentiment words and then input 'GOODREVIEW' or 'BADREVIEW'
    into the review for the LDA to understand topic context"""

    print 'Analyzing Reviews...'


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


    texts = []
    for i in Reviews:
    # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        texts.append(tokens)

    editedT=[]
    for line in texts :
        newline=[]
        for aword in line :
            if aword in pos : 
                newline.append(aword)
                newline.append('GOODREVIEW')
            elif aword in neg :
                newline.append(aword)
                newline.append('BADREVIEW')
            else : newline.append(aword)
        A = ' '.join(word for word in newline)
        editedT.append(A)
    return editedT

######################################
#    Step 3: Finding Topics Reviews
######################################

def LDA_Analysis(editedT=[]):
    """ Perform the LDA analysis on the prepared review vector """

    print 'Finding topics...'

    tdm = textmining.TermDocumentMatrix()


    for doc in editedT:
        tdm.add_doc(doc)

    temp = list(tdm.rows(cutoff=2))


# get the vocab from first row
    vocab = tuple(temp[0])

    Xtest = np.array(temp[1:])

    
    f=file('ldamodel30.pkl','rb')
    R=pickle.load(f)
    L=R.transform(Xtest)

    doc_topic=np.mean(L, axis=0)
    GoodReviews = doc_topic[0]
    BadDurability = doc_topic[2]
    BadCableCord = doc_topic[3]
    GoodCableCord = doc_topic[4]
    GoodHandsFree = doc_topic[6]
    BadReviews = doc_topic[7]
    GoodLevels = doc_topic[8]
    GoodBrand = doc_topic[11]+doc_topic[14]+doc_topic[26]
    GoodComfort = doc_topic[12]
    GoodDurability = doc_topic[15]
    BadService = doc_topic[16]
    BadHeadsetWireless = doc_topic[17]
    GoodSound = doc_topic[18]
    BadComfort = doc_topic[19]+doc_topic[21]
    GoodCase = doc_topic[22]
    BadValue = doc_topic[23]
    GoodValue = doc_topic[24]+doc_topic[25]
    GoodMic = doc_topic[27]
    BadLevels = doc_topic[29]

    datap =   [GoodReviews,GoodCableCord,GoodHandsFree,GoodLevels,GoodComfort,
                GoodDurability,GoodSound,GoodCase,GoodMic,GoodValue,
                BadValue,BadDurability,BadCableCord,BadReviews,BadService,BadHeadsetWireless,
                BadComfort,BadLevels]

    #PLOT_PRODUCT_TOPIC(datap,1)

    print datap

    response=Check_variable_response(datap)
    labels = ['Good Reviews','Good Cable/Cord','Good Handsfree/Wireless', 'Good Levels','Good Comfort/Fit',
              'Good Durability','Good Sound','Good Case','Good Mic','Good Value','Bad Value','Bad Durability',
              'Bad Cable/Cord','Bad Reviews','Bad Customer Service','Bad Handsfree/Wireless','Bad Comfort/Fit',
              'Bad Sound']
    LabeledResponse=np.array([response,labels]).T
    ind=(-np.array(response)).argsort()
    SortedResponse=LabeledResponse[ind]

    return(datap,SortedResponse)

####################################################
#    Step 4: Finding Customized Improvement Strategy
####################################################

def Calculate_Improvement(datap=[],STAR=0):
    """Using the product topic vector find the nearest point in topic space to hypothetical 'goal' products"""


    
    if STAR <3.3 : 
        GoalStars=4.5
    if STAR <2.8 : 
        GoalStars=4.0
    if STAR >=3.3 : 
        GoalStars=5.0
#y=np.load('yfinal.npy')



    print 'Calculating a customized improvement strategy for your product with ',STAR,'Stars to become a product with ',GoalStars,'Stars'


    print datap

    improvements=find_best_shoot(datap,GoalStars,10000)
    
    darrayGood=improvements[0:10]
    darrayBad=improvements[10:]
    print 'Done'
    #PLOT_PRODUCT_TOPIC(improvements,2)
    return darrayGood, -darrayBad






