from flask import render_template
from flaskexample import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
from flask import request

from a_Model import ModelIt

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

import flask as flsk

import sys

#import matplotlib.pyplot as plt
from SegmentedFrontEnd import *

from plotly.offline import plot
from plotly.graph_objs import Scatter
from plotly.graph_objs import Bar
from plotly.graph_objs import Layout
from plotly.graph_objs import Figure
from plotly.graph_objs import Margin
from flask import Markup


@app.route('/index')
def index():
	return render_template("index.html")


@app.route('/input')
def cesareans_input():
    return render_template("input.html")

@app.route('/indexOut')
def indexOut():
  
  url_input= str(request.args.get('user_product_link'))
  #time.sleep(5)
  Reviews_collection,Prod_Star,Prod_Name=Collect_Reviews(url_input)
  editedT=Sentimentize_Reviews(Reviews_collection)
  datav,sortedresponse=LDA_Analysis(editedT)
  darrayG,darrayB=Calculate_Improvement(datav,Prod_Star)
  trace0 = Bar(
  	x=['Good Reviews','Good Cable/Cord','Good Handsfree/Wireless', 'Good Levels','Good Comfort/Fit',
              'Good Durability','Good Sound','Good Case','Good Mic','Good Value','Bad Value','Bad Durability',
              'Bad Cable/Cord','Bad Reviews','Bad Customer Service','Bad Handsfree/Wireless','Bad Comfort/Fit',
              'Bad Sound'],
    y=datav,
    marker=dict(
        color=['rgba(63, 191, 127, 1)', 'rgba(63, 191, 127, 1)','rgba(63, 191, 127, 1)',
        'rgba(63, 191, 127, 1)','rgba(63, 191, 127, 1)','rgba(63, 191, 127, 1)',
        'rgba(63, 191, 127, 1)','rgba(63, 191, 127, 1)','rgba(63, 191, 127, 1)','rgba(63, 191, 127, 1)',
        'rgba(222,45,38,0.8)','rgba(222,45,38,0.8)','rgba(222,45,38,0.8)','rgba(222,45,38,0.8)',
        'rgba(222,45,38,0.8)','rgba(222,45,38,0.8)','rgba(222,45,38,0.8)','rgba(222,45,38,0.8)'
               ]),
  )

  data1 = [trace0]
  layout1 = Layout(
    title='Product Breakdown',
  )

  fig1 = Figure(data=data1, layout=layout1)

  my_plot_div1 = plot(fig1,output_type='div')

  trace2 = Bar(
  	x=['Good Reviews','Good Cable/Cord','Good Handsfree/Wireless', 'Good Levels','Good Comfort/Fit',
              'Good Durability','Good Sound','Good Case','Good Mic','Good Value'],
    y=darrayG,
    marker=dict(
        color=['rgba(63, 191, 127, 1)', 'rgba(63, 191, 127, 1)','rgba(63, 191, 127, 1)',
        'rgba(63, 191, 127, 1)','rgba(63, 191, 127, 1)','rgba(63, 191, 127, 1)',
        'rgba(63, 191, 127, 1)','rgba(63, 191, 127, 1)','rgba(63, 191, 127, 1)','rgba(63, 191, 127, 1)'
               ]),
  )

  data2 = [trace2]
  layout2 = Layout(
    title='Topics to Improve by',
  )

  fig2 = Figure(data=data2, layout=layout2)

  my_plot_div2 = plot(fig2,output_type='div')

  trace3 = Bar(
  	x=['Bad Value','Bad Durability',
              'Bad Cable/Cord','Bad Reviews','Bad Customer Service','Bad Handsfree/Wireless','Bad Comfort/Fit',
              'Bad Sound'],
    y=darrayB,
    marker=dict(
        color=['rgba(222,45,38,0.8)','rgba(222,45,38,0.8)','rgba(222,45,38,0.8)','rgba(222,45,38,0.8)',
        'rgba(222,45,38,0.8)','rgba(222,45,38,0.8)','rgba(222,45,38,0.8)','rgba(222,45,38,0.8)'
                  ]),
  )

  data3 = [trace3]
  layout3 = Layout(
    title='Topics to Reduce by',
  )

  fig3 = Figure(data=data3, layout=layout3)

  my_plot_div3 = plot(fig3,output_type='div')


  trace4 = Bar(
  	x=sortedresponse[:,0],
    y=sortedresponse[:,1],
    orientation = 'h')
  

  data4 = [trace4]
  layout4 = Layout(
    title='Topics that contribute most to Star score',
    height=500,
    margin=Margin(
        l=200,
        r=50,
        b=100,
        t=100)
    
  )

  fig4 = Figure(data=data4, layout=layout4)

  my_plot_div4 = plot(fig4,output_type='div')





  the_result = Prod_Name
  
  #the_result = darray
  return render_template("indexOut.html", the_result = the_result, 
  	the_name=Prod_Name,div_placeholder3=Markup(my_plot_div3),div_placeholder4=Markup(my_plot_div4),
  	div_placeholder=Markup(my_plot_div1),div_placeholder2=Markup(my_plot_div2))







