"""
this file cleans the tweet data and creates features for the models

Ben Branchflower
20 Aug 2019
"""

import re

import pandas as pd
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
# constants
SEED = 123
TEST_SIZE = .3

# reading in the data
obama = pd.read_csv('obama_tweets.csv')
trump = pd.read_csv('trump_tweets.txt', sep='|', error_bad_lines=False).reset_index()

# adding obama indicator
obama['obama'] = 1
trump['obama'] = 0

# renaming names consistent for concat
obama.rename(columns={'Text':'text'}, inplace=True)
trump.rename(columns={'index':'text'}, inplace=True)

# concatenating data
both = pd.concat((obama.loc[:,['text','obama']], trump.loc[:,['text','obama']]))

#dropping retweets
both = both.loc[~both.text.str.contains('^RT'),:]

# cleaning text
both.text = both.text.str.strip().str.replace('\s+',' ').str.replace('(?:: )?https?://.+(?:\s|$)','').str.lower()

# generating feature variables
both['tokens'] = [TweetTokenizer().tokenize(x) for x in both.text]
both['total_words'] = [len(x) for x in both.tokens]
both['avg_word_len'] = [sum([len(y) for y in x]) / len(x) for x in both.tokens]
both['n_cap_let'] = [len(re.findall('[A-Z]',x)) for x in both.text]

# parsing out words
''' this part filters out stopwords like 'and', 'the' and such stopwords needs to be downloaded from nltk to use it
both['no_stop_tokens'] = [x for x in both.tokens if x not in stopwords.words('english')]
both['n_stop_word'] = [x - len(y) for x, y in zip(both.total_xxwords, both.no_stop_tokens)]
''' # Count vectorizer removes stopwords so this is unnecessary unless we wnat stop word counts

# making n_grams
both['bigrams'] = [('_'.join(y) for y in ngrams(x, 2)) for x in both.tokens]
both['trigrams'] = [('_'.join(y) for y in ngrams(x, 3)) for x in both.tokens]

# bag of words features
ngram_tokens = [f"{' '.join(x)} {' '.join(y)} {' '.join(z)}" for x,y,z in zip(both.tokens, both.bigrams, both.trigrams)]
bow_mat = CountVectorizer(max_df=.8, min_df=5).fit_transform(ngram_tokens)
print(bow_mat.shape)

# write the data
feats = both.merge(pd.DataFrame(bow_mat), left_index=True, right_index=True)
feats.to_csv('tweet_feats.txt',sep='|')

# split data into train and test
x_train, x_test, y_train, y_test = train_test_split(feats.loc[:,[x for x in feats.columns if x != 'obama'], feats.obama], random_state=SEEDs)

# modelling
dt = DecisionTreeClassifier(crandom_state=SEED)
