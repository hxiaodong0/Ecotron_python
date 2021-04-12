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
# translator.detect('hello')[0] == 'en'

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

def compare_weights(weights_relavo, weights):
    """
    This function returns the weights relative to the Relavo patent
    :rtype: dictionary
    """
    n = 0
    for key,val in weights.items():
        if key in weights_relavo:
            n += round(val * weights_relavo[key], 2)
    return n

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


file = pd.ExcelFile('Em.xlsx');
print("Loading file successful, the sheets names are: ", file.sheet_names)
df = file.parse("Sheet1")
df["person_name"] = 'nan'
df["company_name"] = 'nan'
df['relevance_socre'] = 'nan'

res = df["company_name"]  # 'result link'

for i in range(res.__len__()):
    person = df['email'][i].split('@')[0]
    website = df['email'][i].split('@')[1]
    df.at[i,'person_name'] = person
    df.at[i,'company_name'] = website

# remove duplicates
df.drop_duplicates(subset ="company_name",
                     keep = False, inplace = True)
df.index = range(df.__len__())
res = df["company_name"]  # 'result link'
len = res.__len__()

key_words = ['Electric' ,'Vehicle','Plug', 'power','Emission','battery', 'Charging','vehicle', 'coach', 'jitney', 'microbus', 'minibus', 'minivan', 'omnibus', 'van'
'convertible', 'fastback', 'hardtop', 'hatchback', 'notchback', 'ragtop', 'sports',
    'car', 'sport' 'vehicle', 'wagon', 'SUV', 'compact', 'coupe', 'coupé', 'intermediate',
             'limousine', 'mini', 'minicar', 'sedan', 'subcompact','hybrid','Battery',
'station','station','EV','Fuel' ,'Cell', 'LEVEL', 'Electric','Electric',
             'Vehicle','Vehicle','Vehicle','car','car','car','car','automobile',
             'automobile','automobile','control','driving','drive','autonomous','transport','transportation' ]

key_words = [x.lower() for x in key_words]
weights_keywords = get_weights(key_words);

# relevance_socre


# http://
# URL = df.company_name[0]

print("being process")

def req(URL):

    # URL =  'https://'+df.company_name[i]

    results = {}

    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36 Edg/89.0.774.68"
    headers = {"user-agent": USER_AGENT}
    n = 0
    try:
        resp = requests.get(URL, headers=headers, verify=False)
        if resp.status_code == 200:
            html = urllib.request.urlopen(URL).read()
            V_text = (text_from_html(html))
            V_text = list(V_text.split(" "))
            V_text = list(filter(None, V_text))

            # if len(V_text) >2:
            #     if translator.detect(V_text[0])[0] != 'en' or translator.detect(V_text[1])[0]!=  'en' :
            #         for i in range(len(V_text)):
            #             V_text[i] = translator.translate(V_text[i], lang_tgt='en')


            weight = get_weights(V_text)
            n = compare_weights(weights_keywords, weight)
    except:
        print("request failed", URL)
        n = -1


    #Tokenize words and create dictionary

    return n

# URL = "https://ecotron.ai/"
# # URL = 'https://' + df.company_name[i]
# n = req(URL)

df = pd.read_pickle("a_file.pkl")

for i in tqdm(range(2726,len)):
    URL = "https://ecotron.ai/"
    URL = 'https://' + df.company_name[i]
    n = req(URL)
    df.at[i, "KeyWord Similarity score"] = n
    # if i %2 == 0:
    df.to_pickle("a_file.pkl")

prep = (
"when", "or", "further", "is", "claim", "wherein", "that", "keep", "causes", "a", "and", "are", "be", "an", "aboard",
"about", "above", "across", "after", "against", "along", "amid", "among", "anti", "around", "as", "at", "before",
"behind", "below", "beneath", "beside", "besides", "between", "beyond", "but", "by", "concerning", "considering"
, "despite", "down", "during", "except", "excepting", "excluding", "following", "for", "from", "in", "inside", "into",
"like", "minus", "near", "of", "off", "on", "onto", "opposite", "outside", "over", "past", "per", "plus", "regarding",
"round", "save", "since"
, "than", "through", "to", "toward", "towards", "under", "underneath", "unlike", "until", "up", "upon", "versus", "via",
"with", "within", "without")
