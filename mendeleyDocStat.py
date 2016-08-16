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
__author__      = "Simon Scheider"
__copyright__   = ""


import plotly
print plotly.__version__




from mendeley import Mendeley
from collections import Counter
import os
import requests
#import pandas

import bibtexparser
import nltk
from nltk.corpus import stopwords
#from wordcloud import WordCloud

s=set(stopwords.words('english'))

txt="a long string of text about him and her"
print filter(lambda w: not w in s,txt.split())

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
    print text
    pass
    # Generate a word cloud image
    #wordcloud = WordCloud().generate(text)

    # Display the generated image:
    # the matplotlib way:
    #import matplotlib.pyplot as plt
    #plt.imshow(wordcloud)
    #plt.axis("off")

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
text =''
n = 0
for i in bib_database.entries:
    #print i
    k = []
    y = ''
    jo = ''
    for j in i.keys():
       # if j == 'title': print (j+ ': ' + i[j])
       # if j == 'url' : print  (j+ ': ' +i[j])
       # if j == 'doi': print  (j+ ': ' +i[j])
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
    tuples.append((k,y,jo))
    n =n+1
    #if n>10: break

wordcl(text)

d = splitByAttribute(tuples,1,0, extend=True)
for i in sorted(d.items()):
    print (i[0] + ': ' +str(stat(i[1])))

dc = sorted(stat(keywords).items(), key=lambda student: student[1], reverse=True)
#print dc
for i in dc:
    if i[1]>10:
        print i

#print years
y = sorted((stat(years)).items(), key=lambda student: student[1], reverse=True)
#print y
for i in y:
    if i[1]>1:
        print i

jou = sorted((stat(journals)).items(), key=lambda student: student[1], reverse=True)
#print y
for i in jou:
    if i[1]>2:
        print i



#doc = session.catalog.by_identifier(doi=doi, view='stats')
#print '"%s" has %s readers.' % (doc.title, doc.reader_count)


def main():
   from plotly.graph_objs import Scatter, Layout
url = plotly.offline.plot({
    "data": [Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1])],
    "layout": Layout(title="hello world")
},filename=r'C:\Users\simon\Dropbox\Tracking technologies paper\figures\temp-plot.html')
print url

if __name__ == '__main__':
    main()
