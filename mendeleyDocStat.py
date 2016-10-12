#-------------------------------------------------------------------------------
# Name:       mendeleyDocStat
# Purpose:      Creates statistics over a mendeley group" research papers
#
# Author:      simon
#
# Created:     12/08/2016
# Copyright:   (c) simon 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from __future__ import print_function
__author__      = "Simon Scheider"
__copyright__   = ""

import sklearn
import numpy
# Import all of the scikit learn stuff

from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
import pandas as pd
import warnings
import string
# Suppress warnings from pandas library
warnings.filterwarnings("ignore", category=DeprecationWarning,
module="pandas", lineno=570)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import plotly


from plotly.graph_objs import Scatter, Bar, Layout

print (plotly.__version__)


from mendeley import Mendeley
from collections import Counter
import os
import requests
#import pandas

import bibtexparser
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud

s=set(stopwords.words('english'))

txt="a long string of text about him and her"
print (filter(lambda w: not w in s,txt.split()))

def hasNumbers(inputString):
     return any(char.isdigit() for char in inputString)

def normalize(text):
    import re
    text = text.lower()
    #t = re.compile(r'\W+', re.UNICODE).split(text)
    #t = filter(lambda w: not w in s,text)
    #print text
    #print t
    tt = []
    t = text.split(',')
    for i in t:
        #print (i + str(hasNumbers(i)))
        if not hasNumbers(i): tt.append(i)
    #
    #for wo in t:
    #    tt.append()
    return tt

def splitByAttribute(items, k, n, extend=False):
    dict = {}
    for i in items:
        if (dict.get(i[k])!=None):
            if extend == False:
                (dict.get(i[k])).append(i[n])
            else:
                (dict.get(i[k])).extend(i[n])
        else:
            if extend == False:
                dict[i[k]]=[i[n]]
            else:
                dict[i[k]]= i[n]
    return dict

def stat(L):
    return Counter(L)


def wordcl(text):
    #print (text)
    #pass
    # Generate a word cloud image
    wordcloud = WordCloud(max_font_size=40, relative_scaling=.2, ranks_only=True, background_color='white').generate(text)

    # Display the generated image:
    # the matplotlib way:
    import matplotlib.pyplot as plt
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.figure()
    plt.show()

    # take relative word frequencies into account, lower max_font_size
    #wordcloud = WordCloud(max_font_size=40, relative_scaling=.5).generate(text)
    #plt.figure()
    #plt.imshow(wordcloud)
    #plt.axis("off")
    #plt.show()

with open('My Collection.bib') as bibtex_file:
    bibtex_str = bibtex_file.read()

bib_database = bibtexparser.loads(bibtex_str)
keywords =[]
years =[]
tuples =[]
journals =[]
abstracts =[]
documents =[]
text =''
n = 0
for i in bib_database.entries:
    #print i
    k = []
    y = ''
    jo = ''
    for j in i.keys():
       ii = {}
       # if j == 'title': print (j+ ': ' + i[j])
       # if j == 'url' : print  (j+ ': ' +i[j])
       # if j == 'doi': print  (j+ ': ' +i[j])
       if j == 'abstract':
            abstracts.append(i[j])
            ii['abstract']=i[j]
       if j == 'title':
            ii['title'] = i[j]
       if j == 'doi'or j == 'url':
            ii['id'] = i[j]
       if j == 'keyword':
            #print  (j+ ': ' +i[j])
            #print normalize(i[j])
            k = normalize(i[j])
            keywords.extend(k)
            for x in k:
                text += (' "'+x+'" ')
                #text += (for x in k:
            #text += (" ".join(str(x) for x in k))
       if j == 'year':
            y =i[j]
            years.append(y)
            #print i[j]
       if j == 'journal':
            jo =i[j]
            journals.append(jo)
       # if j == 'author': print  (j+ ': ' +i[j])
       documents.append(ii)
    tuples.append((k,y,jo))
    n =n+1
    #if n>10: break

#wordcl(text)
#print (pd.DataFrame(documents))

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    tokens = [i for i in tokens if i not in string.punctuation and len(i)>=3]
    tokens = [i for i in tokens if i.isalpha()]
    #stems = stem_tokens(tokens, stemmer)
    return tokens

def display(data, labels, xname, yname, zname):
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=20, azim=160)

    plt.cla()

    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels.astype(numpy.float))
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_zlabel(zname)
    plt.show()

def getClusterText(labels, texts):
    d = {}
    if len(labels)!= len(texts):
        return
    for i in labels:
        if i not in d.keys():
            d[i]= texts[i]
        else:
            d[i]=(d[i])+ ' '+texts[i]
    return d

def generateWordCloudsforClusters(cltexts):
    for i in cltexts.keys():
        wordcl(cltexts[i])




vectorizer = TfidfVectorizer(min_df = 1, stop_words = 'english', analyzer = 'word', tokenizer=tokenize)

dtm = vectorizer.fit_transform(abstracts)
print (len(vectorizer.get_feature_names()))
#print (pd.DataFrame(dtm.toarray(),index=abstracts,columns=vectorizer.get_feature_names()).head(10))
lsa = TruncatedSVD(3, algorithm = 'arpack')
dtm_lsa = lsa.fit_transform(dtm)
dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)
print (pd.DataFrame(lsa.components_,index = ["component_1","component_2", "component_3"],columns = vectorizer.get_feature_names()))
#documentword = numpy.asarray(numpy.asmatrix(dtm_lsa)*numpy.asmatrix(lsa.components_))
#pd.DataFrame(documentword, columns=vectorizer.get_feature_names()))
#print (documentword[0])
print (dtm_lsa)
print (len(dtm_lsa))

kmeans = KMeans(n_clusters=5).fit_predict(numpy.array(dtm_lsa))
display(dtm_lsa, kmeans,"component_1","component_2", "component_3")

t = getClusterText(kmeans,abstracts)

generateWordCloudsforClusters(t)


d = splitByAttribute(tuples,1,0, extend=True)
for i in sorted(d.items()):
    pass
    #print (i[0] + ': ' +str(stat(i[1])))

x = []
y = []

dc = sorted(stat(keywords).items(), key=lambda student: student[1], reverse=True)
#print dc
for i in dc:
    if i[1]>10:
        #print (i)
        if (i[0]): x.append(i[0])
        else: x.append(None)
        if (i[1]): y.append(i[1])
        else: y.append(None)

##keyws = plotly.offline.plot({
##    "data": [Bar(x=x, y=y)],
##    "layout": Layout(title="Keyword frequency")
##    },filename=r'C:\Users\simon\Dropbox\Tracking technologies paper\figures\keyws.html')
###print (keyws)
x = []
y = []
#print years
yearplot = sorted((stat(years)).items(), key=lambda student: student[1], reverse=True)
#print y
for i in yearplot:
    if i[1]>1:
        #print (i)
        if (i[0]): x.append(i[0])
        else: x.append(None)
        if (i[1]): y.append(i[1])
        else: y.append(None)

##artcls = plotly.offline.plot({
##    "data": [Bar(x=x, y=y)],
##    "layout": Layout(title="Number of Articles per year")
##    },filename=r'C:\Users\simon\Dropbox\Tracking technologies paper\figures\articlesperyear.html')
###print (artcls)
x = []
y = []

jou = sorted((stat(journals)).items(), key=lambda student: student[1], reverse=True)
#print y
for i in jou:
    if i[1]>2:
        #print (i)
        if (i[0]): x.append(i[0])
        else: x.append(None)
        if (i[1]): y.append(i[1])
        else: y.append(None)
##journls = plotly.offline.plot({
##    "data": [Bar(x=x, y=y)],
##    "layout": Layout(title="Journal frequency")
##    },filename=r'C:\Users\simon\Dropbox\Tracking technologies paper\figures\journalfrequency.html')
###print (journls)
x = []
y = []


#doc = session.catalog.by_identifier(doi=doi, view='stats')
#print '"%s" has %s readers.' % (doc.title, doc.reader_count)


def main():
    pass



if __name__ == '__main__':
    main()
