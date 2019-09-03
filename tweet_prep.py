"""
this file cleans the tweet data and creates features for the models

Ben Branchflower
20 Aug 2019
"""

import json
import re

import joblib
import pandas as pd
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
from mlxtend.plotting import plot_decision_regions
from matplotlib import pyplot as plt

# constants
SEED = 123
TEST_SIZE = .05  # the proportion of data left out of sample
# MAX_FEATURES = 7500 # the maximum number of bag of words features
N_JOBS = 11  # the number of threads to be used during training, -1 uses all processors available
DEPTH = 3  # the depth of visualization of decision tree
SVM_FEATS = 1000


def text_to_features(tweet_text):
    '''
    This function takes the raw text from a tweet and prepares it to be
    classified by generating the numeric features used in training
    arguments:
        tweet_text (str) - the text from a tweet
        terms (list like) - the words/ngrams used in the bag of words features
    returns:
        array of numeric features
    '''
    tweet_text = re.sub(r'[…"#$%&\'\(\)*+,-./:;<=>?@\[\\\]^_`{|}~’“”—]', '', tweet_text)
    tweet_text = re.sub('–|––|—|\s+', ' ', tweet_text)
    tknzr = TweetTokenizer()
    tokens = tknzr.tokenize(tweet_text)
    ngram_tokens = tokens.copy()
    n_tokens = len(tokens)
    avg_tok_len = sum([len(x) for x in tokens]) / n_tokens
    '''
    ngram_tokens += [re.sub('^_|\s_|_\s','','_'.join(y)) for y in ngrams(tokens, 2)]
    ngram_tokens += [re.sub('^_|\s_|_\s','','_'.join(y)) for y in ngrams(tokens, 3)]
    token_text = re.sub('\b_|_\b','',' '.join(ngram_tokens)).lower()
    print(token_text)
    '''
    bow_vec = vectorizer.transform([tweet_text])
    feats = pd.DataFrame(bow_vec.A, columns=vectorizer.get_feature_names())
    feats['total_words'] = [n_tokens]
    feats['n_cap_let'] = [len(re.findall('[A-Z]', tweet_text))]
    feats['avg_word_len'] = [avg_tok_len]
    return feats


def model_tuning(model, params, outfile=None, n_folds=5):
    '''
    this function runs a gridsearch for the given parameters
    it is just a couple lines but I will be using it enough it seems worth it
    '''
    gridsearch = GridSearchCV(model, params, cv=n_folds)
    gridsearch.fit(x_train, y_train)
    print('Best Parameters', gridsearch.best_params_)
    print('Train time', gridsearch.refit_time_)
    print(classification_report(y_test, gridsearch.best_estimator_.predict(x_test)))
    if outfile is not None:
        joblib.dump(gridsearch, outfile)
    return gridsearch.best_estimator_


if __name__ == '__main__':
    # reading in the data
    obama = pd.read_csv('obama_tweets.csv')
    trump = pd.DataFrame(json.load(open('trump_tweets.json')))
    tknzr = TweetTokenizer()

    print('read in the data...')

    # adding obama indicator
    obama['obama_indicator'] = 1
    trump['obama_indicator'] = 0

    # renaming names consistent for concat
    obama.rename(columns={'Text': 'text'}, inplace=True)

    # concatenating data
    both = pd.concat((obama.loc[:, ['text', 'obama_indicator']], trump.loc[:, ['text', 'obama_indicator']]))

    both.rename(columns={'text': '_text'}, inplace=True)
    # dropping retweets
    both = both.loc[~both._text.str.contains('^RT'), :]
    print('Shape of raw data:', both.shape)

    # cleaning text
    both._text = both._text.str.strip().str.replace('\s+', ' ').str.replace('(?:: )?https?://.+(?:\s|$)', '')
    both._text = both._text.str.replace(r'\.?pic\.twitter\.com/.+(?:\s|$)', '')
    both._text = both._text.str.replace(r'\d+', '')

    print('data cleaned...')

    both._text = both._text.str.replace(r'[…"#$%&\'\(\)*+,-./:;<=>?@\[\\\]^_`{|}~’“”—]', '')
    both._text = both._text.str.replace('–|––|\s+', ' ')
    both['n_cap_let'] = [len(re.findall('[A-Z]', x)) for x in both._text]
    both._text = both._text.str.lower()

    # parsing text and dropping empty tweets after cleaning
    both['tokens'] = [[re.sub('_', '', y) for y in tknzr.tokenize(x)] for x in both._text]
    both = both.loc[[len(x) > 0 for x in both.tokens], :]

    # generating feature variables
    both['total_words'] = [len(x) for x in both.tokens]
    both['avg_word_len'] = [sum([len(y) for y in x]) / len(x) for x in both.tokens]

    # parsing out words
    ''' this part filters out stopwords like 'and', 'the' and such stopwords needs to be downloaded from nltk to use it
    both['no_stop_tokens'] = [x for x in both.tokens if x not in stopwords.words('english')]
    both['n_stop_word'] = [x - len(y) for x, y in zip(both.total_words, both.no_stop_tokens)]
    '''  # Count vectorizer removes stopwords so this is unnecessary unless we want stop word counts

    # making n_grams
    '''
    both['bigrams'] = [(re.sub('^_|\s_|_\s|_$','','_'.join(y)) for y in ngrams(x, 2)) for x in both.tokens]
    both['trigrams'] = [(re.sub('^_|\s_|_\s|_$','','_'.join(y)) for y in ngrams(x, 3)) for x in both.tokens]
    '''
    print('preliminary features complete...')

    # bag of words features
    '''
    ngram_tokens = [f"{' '.join(x)} {' '.join(y)} {' '.join(z)}" for x,y,z in zip(both.tokens, both.bigrams, both.trigrams)]
    ngram_tokens = [re.sub('(?:\s|^)_|_(?:\s|$)','',x) for x in ngram_tokens]
    '''
    vectorizer = CountVectorizer(max_df=.5, min_df=.0001, stop_words='english', tokenizer=tknzr.tokenize,
                                 ngram_range=(1, 3))
    bow_mat = vectorizer.fit_transform(both._text)
    print('Bag of words feature set:', bow_mat.shape)

    # write the data
    feats = both.merge(pd.DataFrame(bow_mat.A, columns=vectorizer.get_feature_names()), left_index=True,
                       right_index=True)
    print(feats.columns)
    feats = feats.loc[:, [x for x in feats.columns if x not in ('_text', 'tokens', 'bigrams', 'trigrams')]]
    feats = feats.loc[:, [x for x in feats.columns if feats[x].sum() > 0]]

    print('Completed Feature construction')
    print('Final size:', feats.shape)
    # feats.to_csv('tweet_feats.txt',sep='|')

    # split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(
        feats.loc[:, [x for x in feats.columns if x != 'obama_indicator']],
        feats.obama_indicator, random_state=SEED)
    print('\nDecision Tree')
    # decision tree
    dtree = DecisionTreeClassifier(max_features=x_train.shape[1],
                                   class_weight='balanced',
                                   min_samples_split=20, random_state=SEED)
    '''
    # visualize decision tree
    dtree = model_tuning(dtree, {'max_depth':[100]}, 'best_dtree.joblib')


    dot_data = tree.export_graphviz(dtree, feature_names=x_train.columns, class_names=['Trump', 'Obama'],
                                    filled=True, rotate=True, max_depth=DEPTH)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png(f'{DEPTH}_level_decision_tree.png')

    # random forest
    print('\nRandom Forest')
    rfc = RandomForestClassifier(n_estimators=1000, max_depth=150,
                                    class_weight='balanced',
                                    random_state=SEED, n_jobs=N_JOBS)

    rfc = model_tuning(rfc, {'max_depth':[100,150]}, 'best_rf.joblib')

    # getting top features to reduce SVM training time
    rfc = joblib.load('best_rf.joblib').best_estimator_
    best_feats = rfc.feature_importances_.argsort()[:-SVM_FEATS][::-1]
    x_train = x_train.iloc[:,best_feats]
    x_test = x_test.iloc[:,best_feats]

    # Support Vector Machine
    print('\nSupport Vector Machine')
    svm = SVC(gamma='scale',class_weight='balanced', random_state=SEED, max_iter=1000)

    for kernel in ('poly','rbf','sigmoid'):
        print(kernel, '...', sep='')
        kwargs = {'kernel':[kernel]}
        if kernel == 'poly':
            kwargs['degree'] = [1,2,3]
            kwargs['coef0'] = [0]
        elif kernel == 'sigmoid':
            kwargs['coef0'] = [0.]
        svm = model_tuning(svm, kwargs, f'best_{kernel}_svm.joblib')
    # Neural Network
    '''
    '''
    # load in the pretrained models
    dtree = joblib.load('best_dtree.joblib').best_estimator_
    rfc = joblib.load('best_rf.joblib').best_estimator_
    svm_poly = joblib.load('best_poly_svm.joblib').best_estimator_

    tweet_text = "Make America Great Again"
    tweet_feats = text_to_features(tweet_text)

    for model in (dtree,rfc,svm_poly):
        print(model.predict(tweet_feats))
        print(model.predict_proba(tweet_feats))
    '''
    ## SVM visualization
    # train a model on two components
    print('SVM visualization')
    X = feats[['total_words', 'n_cap_let']]
    y = feats['obama_indicator']

    svm = SVC(random_state=SEED, kernel='poly', degree=3, gamma='scale',max_iter=1500)
    svm.fit(X.values, y.values)
    plot_decision_regions(X.values, y.values, clf=svm, legend=2)

    plt.xlabel('Total Words')
    plt.ylabel('Number of Capital Letters')
    plt.title('SVM Decision Regions')
    plt.savefig('SVM_viz_ncap_on_total.png')