#Running time approximately: <1hour
#reference sites https://dev.to/coderasha/compare-documents-similarity-using-python-nlp-4odp
import string
import nltk
import gensim
import numpy as np
nltk.download('punkt')
from nltk.tokenize import word_tokenize , sent_tokenize
from bs4 import BeautifulSoup
import requests
import pandas as pd
from tqdm import tqdm
import time
import googletrans
from googletrans import Translator
from google_trans_new import google_translator
from googletrans import Translator
from bs4.element import Comment
import urllib.request

translator = google_translator()
translate_text = translator.translate('สวัสดีจีน',lang_tgt='en')
print(translate_text)
# translator.detect('hello')[0] == 'en'



def get_weights(key_words ):
    gen_docs = [[w.lower() for w in word_tokenize(text)]
                for text in key_words]


    dictionary = gensim.corpora.Dictionary(gen_docs)
    # print(dictionary.token2id)
    # create a bag of words
    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
    # print(corpus)
    # print(corpus)

    # TFIDF Term Frequency – Inverse Document Frequency(TF-IDF)
    # words that occur more frequently across the documents get bigger weights.
    tf_idf = gensim.models.TfidfModel(corpus)
    lst = []
    for doc in tf_idf[corpus]:
        lst.append([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])
        # print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])

    dic = {}
    for item in lst:
        for i in item:
            if i[0] not in dic.keys():
                dic[i[0]] = i[1]
            else:
                dic[i[0]] = round(dic[i[0]] + i[1], 2)

    dic_copy = dic.copy()
    for key, val in dic_copy.items():
        try:
            (int(key))
            dic.pop(key)
        except:
            continue
    return dic


print("hello world")
file = pd.ExcelFile('Em.xlsx');
print("Loading file successful, the sheets names are: ", file.sheet_names)
df = file.parse("Sheet1")
df["person_name"] = 'nan'
df["company_name"] = 'nan'

res = df["company_name"]  # 'result link'

for i in range(res.__len__()):
    person = df['email'][i].split('@')[0]
    website = df['email'][i].split('@')[1]
    df.at[i,'person_name'] = person
    df.at[i,'company_name'] = website

# remove duplicates
df.drop_duplicates(subset ="company_name",
                     keep = False, inplace = True)
len = res.__len__()

key_words = ['Electric' ,'Vehicle','Plug', 'power','Emission','battery', 'Charging','vehicle', 'coach', 'jitney', 'microbus', 'minibus', 'minivan', 'omnibus', 'van'
'convertible', 'fastback', 'hardtop', 'hatchback', 'notchback', 'ragtop', 'sports',
    'car', 'sport' 'vehicle', 'wagon', 'SUV', 'compact', 'coupe', 'coupé', 'intermediate',
             'limousine', 'mini', 'minicar', 'sedan', 'subcompact','hybrid','Battery',
'station','station','EV','Fuel' ,'Cell', 'LEVEL', 'Electric','Electric',
             'Vehicle','Vehicle','Vehicle','car','car','car','car','automobile',
             'automobile','automobile','control','driving','drive','autonomous' ]

key_words = [x.lower() for x in key_words]
weights = get_weights(key_words);


USER_AGENT  = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36"
# http://
# URL = df.company_name[0]

URL = "https://2ndsightbio.com/"

results = {}


headers = {"user-agent": USER_AGENT}

try:
    resp = requests.get(URL, headers=headers)
except:
    print("request failed", URL)

#Tokenize words and create dictionary

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return u" ".join(t.strip() for t in visible_texts)

html = urllib.request.urlopen('https://2ndsightbio.com/').read()
V_text = (text_from_html(html))
V_text = list(V_text.split(" "))

prep = (
"when", "or", "further", "is", "claim", "wherein", "that", "keep", "causes", "a", "and", "are", "be", "an", "aboard",
"about", "above", "across", "after", "against", "along", "amid", "among", "anti", "around", "as", "at", "before",
"behind", "below", "beneath", "beside", "besides", "between", "beyond", "but", "by", "concerning", "considering"
, "despite", "down", "during", "except", "excepting", "excluding", "following", "for", "from", "in", "inside", "into",
"like", "minus", "near", "of", "off", "on", "onto", "opposite", "outside", "over", "past", "per", "plus", "regarding",
"round", "save", "since"
, "than", "through", "to", "toward", "towards", "under", "underneath", "unlike", "until", "up", "upon", "versus", "via",
"with", "within", "without")
