"""
this file cleans the tweet data and creates features for the models

Ben Branchflower
20 Aug 2019
"""

import json
import re
from collections import Counter

import pandas as pd
import pickle
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn import tree
import pydotplus

# constants
SEED = 123
TEST_SIZE = .3

def text_to_features(tweet_text, terms):
    '''
    This function takes the raw text from a tweet and prepares it to be
    classified by generating the numeric features used in training
    arguments:
        tweet_text (str) - the text from a tweet
        terms (list like) - the words/ngrams used in the bag of words features
    returns:
        array of numeric features
    '''
    tknzr = TweetTokenizer()
    tokens = tknzr.tokenize(tweet_text)
    n_tokens = len(tokens)
    avg_tok_len = sum([len(x) for x in tokens]) / n_tokens
    tokens += [re.sub('^_|\s_|_\s','','_'.join(y)) for y in ngrams(tokens, 2)]
    tokens += [re.sub('^_|\s_|_\s','','_'.join(y)) for y in ngrams(tokens, 3)]
    counts = Counter(tokens)
    feats = {x:counts.get(x,0) for x in terms}
    feats['total_words'] = n_tokens
    feats['n_cap_let'] = len(re.findall('[A-Z]',tweet_text))
    feats['avg_word_len'] = avg_tok_len
    return pd.Series(feats)


def model_tuning(model, params, outfile=None):
    '''
    this file runs a gridsearch for the given parameters
    it is just a couple lines but I will be using it enough it seems worth it
    '''
    gridsearch = GridSearchCV(model, params)
    gridsearch.fit()
    print('Best Parameters', gridsearch.best_params_)
    print('Train time', gridsearch.refit_time_)
    print(classification_report(y_test, gridsearch.best_model_.predict(x_test)))
    if outfile is not None:
        pickle.dump(gridsearch, outfile)
    return gridsearch.best_model_


if __name__ == '__main__':
    # reading in the data
    obama = pd.read_csv('obama_tweets.csv')
    trump = pd.DataFrame(json.load(open('trump_tweets.json')))

    print('read in the data...')

    # adding obama indicator
    obama['obama_indicator'] = 1
    trump['obama_indicator'] = 0

    # renaming names consistent for concat
    obama.rename(columns={'Text':'text'}, inplace=True)

    # concatenating data
    both = pd.concat((obama.loc[:,['text','obama_indicator']], trump.loc[:,['text','obama_indicator']]))

    #dropping retweets
    both = both.loc[~both.text.str.contains('^RT'),:]

    # cleaning text
    both.text = both.text.str.strip().str.replace('\s+',' ').str.replace('(?:: )?https?://.+(?:\s|$)','')
    both.text = both.text.str.replace(r'\.?pic\.twitter\.com/.+(?:\s|$)','')
    both.text = both.text.str.replace(r'\d+','')

    print('data cleaned...')

    both['n_cap_let'] = [len(re.findall('[A-Z]',x)) for x in both.text]
    both.text = both.text.str.lower()

    # parsing text and dropping empty tweets after cleaning
    both['tokens'] = [[re.sub('_', '', y) for y in TweetTokenizer().tokenize(x)] for x in both.text]
    both = both.loc[[len(x) > 0 for x in both.tokens],:]

    # generating feature variables
    both['total_words'] = [len(x) for x in both.tokens]
    both['avg_word_len'] = [sum([len(y) for y in x]) / len(x) for x in both.tokens]


    # parsing out words
    ''' this part filters out stopwords like 'and', 'the' and such stopwords needs to be downloaded from nltk to use it
    both['no_stop_tokens'] = [x for x in both.tokens if x not in stopwords.words('english')]
    both['n_stop_word'] = [x - len(y) for x, y in zip(both.total_words, both.no_stop_tokens)]
    ''' # Count vectorizer removes stopwords so this is unnecessary unless we want stop word counts

    # making n_grams
    both['bigrams'] = [(re.sub('^_|\s_|_\s','','_'.join(y)) for y in ngrams(x, 2)) for x in both.tokens]
    both['trigrams'] = [(re.sub('^_|\s_|_\s','','_'.join(y)) for y in ngrams(x, 3)) for x in both.tokens]

    print('preliminary features complete...')

    # bag of words features
    ngram_tokens = [f"{' '.join(x)} {' '.join(y)} {' '.join(z)}" for x,y,z in zip(both.tokens, both.bigrams, both.trigrams)]
    ngram_tokens = [re.sub('(?:\s|^)_|_(?:\s|$)','',x) for x in ngram_tokens]
    vectorizer = CountVectorizer(max_df=.8, min_df=15, max_features=5000)
    bow_mat = vectorizer.fit_transform(ngram_tokens)
    print('Bag of words feature set:', bow_mat.shape)

    # write the data
    feats = both.merge(pd.DataFrame(bow_mat.A, columns=vectorizer.get_feature_names()), left_index=True, right_index=True)
    feats = feats.loc[:,[x for x in feats.columns if x not in ('text','tokens','bigrams','trigrams')]]

    print('Completed Feature construction')
    # feats.to_csv('tweet_feats.txt',sep='|')

    # split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(feats.loc[:,[x for x in feats.columns if x != 'obama_indicator']],
                                                        feats.obama_indicator, random_state=SEED)

    # modelling
    dtree = DecisionTreeClassifier(max_depth=3, max_features=5003, class_weight='balanced',
                                    random_state=SEED)
    dtree.fit(x_train, y_train)

    # visualize decision tree
    dot_data = tree.export_graphviz(dtree, feature_names=x_train.columns)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('3_level_decision_tree.png')

    dtree = model_tuning(dtree, {'max_depth':[100]})
