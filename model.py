import numpy as np
import pandas as pd
import pickle
import joblib
data = pd.read_csv("JokeDetectionDataset_classified.csv")
data.head()
import spacy
nlp = spacy.load("en_core_web_sm")
spacy.explain('PROPN')
humor_texts = data[data['humor'] == True]['text']
non_humor_texts = data[data['humor'] == False]['text']
humor_word_counts = [len(text.split()) for text in humor_texts]
non_humor_word_counts = [len(text.split()) for text in non_humor_texts]
humor_char_counts = [len(text) for text in humor_texts]
non_humor_char_counts = [len(text) for text in non_humor_texts]
qm_jokes = 0
qm_no_jokes = 0

for text in data[data.humor == False]['text']:
    if '?' in text:
        qm_no_jokes += 1

for text in data[data.humor == True]['text']:
    if '?' in text:
        qm_jokes += 1

df_qm = pd.DataFrame({'status':['A Joke', 'Not A Joke'], 'question_mark':[qm_jokes, qm_no_jokes]})
from spacy.matcher import Matcher
from tqdm.auto import tqdm

puncs = []
length = []
ratio = []

matcher = Matcher(nlp.vocab)
punc_pattern = [{'POS':'PUNCT'}]
matcher.add(1, [punc_pattern])

progress_bar = tqdm(range(len(data)))

for text in data['text']:
    doc = nlp(text)
    matches = matcher(doc)

    l_app = len(doc)
    p_app = len(matches)
    r_app = p_app / l_app

    length.append(l_app)
    puncs.append(p_app)
    ratio.append(r_app)

    progress_bar.update(1)
df_ratio = pd.DataFrame({'humor':data.humor, 'ratio':ratio})
df_ratio_counts = pd.DataFrame({'humor':[True, False],
                                'ratio':[df_ratio[df_ratio.humor == x].ratio.mean() for x in [True, False]]})
from spacy.matcher import Matcher
from tqdm.auto import tqdm

prpn = []
adj = []
noun = []
verb = []
length = []

prpn_pattern = [{'POS':'PROPN'}]
adj_pattern = [{'POS':'ADJ'}]
noun_pattern = [{'POS':'NOUN'}]
verb_pattern = [{'POS':'VERB'}]

matcher_propn = Matcher(nlp.vocab)
matcher_adj = Matcher(nlp.vocab)
matcher_noun = Matcher(nlp.vocab)
matcher_verb = Matcher(nlp.vocab)

matcher_propn.add(1, [prpn_pattern])
matcher_adj.add(1, [adj_pattern])
matcher_noun.add(1, [noun_pattern])
matcher_verb.add(1, [verb_pattern])

progress_bar = tqdm(range(len(data)))

for text in data['text']:
    doc = nlp(text)

    prpn.append(len(matcher_propn(doc)))
    adj.append(len(matcher_adj(doc)))
    noun.append(len(matcher_noun(doc)))
    verb.append(len(matcher_verb(doc)))
    length.append(len(doc))

    progress_bar.update(1)
df_pos = pd.DataFrame({'propn':np.array(prpn)/np.array(length),
                       'adj':np.array(adj)/np.array(length),
                       'noun':np.array(noun)/np.array(length),
                       'verb':np.array(verb)/np.array(length),
                       'length':length,
                       'humor':data.humor})
pos = ['propn', 'adj', 'noun', 'verb']
pos_count = {}
for p in pos:
    pos_count[p] = [df_pos[df_pos.humor == x][p].mean() for x in [True, False]]

df_pos_count = pd.DataFrame(pos_count)
df_pos_count['humor'] = [True, False]
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
X = data['text']
y = data['humor']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vec, y_train)
pickle.dump(nb_classifier, open('model.pkl','wb'))
joblib.dump(vectorizer,'vect.pkl')