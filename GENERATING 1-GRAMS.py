#IMPORTING LIBRARIES////////////////////////////////
import io
import re
import json
import pandas as pd
import numpy as np
import operator
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from collections import Counter
from textblob import TextBlob
from string import digits
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns

#READING THE INPUT DATASET////////////////////////////////
df = pd.read_csv('DATA.csv', index_col=None,encoding= 'unicode_escape' )

#PRE PROSSEING THE TEXT ////////////////////////////////
df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))#CHANGING COLUMN NAMED "text" TO LOWERCASE
df['text'] = df['text'].apply(lambda x: " ".join(x.replace("'", "") for x in x.split()))#REMOVING SINGLE QUOTE
df['text'] = df['text'].str.replace('[^\w\s]','')#REMOVING PUNCTUATIONS
freq = pd.Series(' '.join(df['text']).split()).value_counts()[:10]#REMOCING COMMON WORDS
freq = list(freq.index)
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
freq = pd.Series(' '.join(df['text']).split()).value_counts()[-10:]#REMOVING RARE WORDS
freq = list(freq.index)
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

#TOKENIZING ////////////////////////////////
def identify_tokens(row):
    text = str(row['text'])
    tokens = nltk.word_tokenize(text)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words
df['words'] = df.apply(identify_tokens, axis=1)

#LEMMATIZING ////////////////////////////////
from textblob import Word
def lemm_list(row):
    my_list = row['words']#over the tokenization
    lemmatized_list = [Word(word).lemmatize()for word in my_list]
    return (lemmatized_list)
df['lemmatized_words'] = df.apply(lemm_list, axis=1)

#PROSSESING DONE ////////////////////////////////
def rejoin_words(row):
    my_list = row['lemmatized_words']
    joined_words = ( " ".join(my_list))
    return joined_words
df['processed'] = df.apply(rejoin_words, axis=1)
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

stopwords = set(STOPWORDS)
stopwords.update(['we','will','aren', 'couldn', 'didn', 'doesn', 'don', 'hadn',
                  'dont','doesnt','cant','couldnt','couldve','im','ive','isnt',
                  'theres','wasnt','wouldnt','a','also','like',
                  'hasn', 'haven', 'isn', 'let', 'll', 'mustn', 're', 'shan', 'shouldn',
                  've', 'wasn', 'weren', 'won', 'wouldn','ha','wa','ldnont'])
                  
#VECTORIZING ////////////////////////////////
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words=stopwords)
bow = bow_vectorizer.fit_transform(df['processed'])
top_sum=bow.toarray().sum(axis=0)
top_sum_cv=[top_sum]
columns_cv = bow_vectorizer.get_feature_names()
x_traincvdf = pd.DataFrame(top_sum_cv,columns=columns_cv)
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000,stop_words=stopwords)
text_content = df['processed']
from nltk.corpus import stopwords

#GENERATING BI-GRAMS AND ONE-GRAMS ////////////////////////////////
stops= ['we','this','at','will','can','be','are','cant','our','on','is','an','are','by','all','it']
text_content = [word for word in text_content if not any(stop in word for stop in stops )]
bigrams_list = list(nltk.bigrams(text_content)) 
dictionary2 = [' '.join(tup) for tup in bigrams_list]
vectorizer = CountVectorizer(ngram_range=(1, 1))#ONE-GRAMS
bag_of_words = vectorizer.fit_transform(dictionary2)
vectorizer.vocabulary_
sum_words = bag_of_words.sum(axis=0) 
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
#print (words_freq[:100])
df2= pd.DataFrame(words_freq[:50])

#WRITING CSV FILE FOR ONE-GRAMS OUT PUT ////////////////////////////////
df2.to_csv('ONE_grams_DATA.csv') 


#REFRENCES:
#1-https://www.kaggle.com/gangakrish/keyword-extraction-from-tweets/notebook
#2-https://www.kaggle.com/amar09/sentiment-analysis-on-scrapped-tweets
