#Running time approximately: < 24hour
#reference sites https://dev.to/coderasha/compare-documents-similarity-using-python-nlp-4odp

#DONE Revision advises: store all the website information is the first step,
# translationAPI implementation;
#DONE long time no response skipping implementation.
#DONE https, http and remove access bar implementation,
# access subpages implementation

# Keyword selection: from Ecotron and key competitors;

import pickle
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
from urllib.parse import urlparse
import http.client, sys
import translators as ts
from itertools import chain
import re
from langdetect import detect
from textblob import TextBlob



def check_url(url):

  try:
    url = urlparse(url)
    conn = http.client.HTTPConnection(url.netloc,timeout=30)
    conn.request("HEAD", url.path)
  except:
      return False
  try:
      if conn.getresponse():
        return True
      else:
        return False
  except:
      return False

def key_words_gen():
    key_words = ['Electric', 'Vehicle', 'Plug', 'power', 'Emission', 'battery', 'Charging', 'vehicle', 'coach',
                 'jitney',
                 'microbus', 'minibus', 'minivan', 'omnibus', 'van'
                                                              'convertible', 'fastback', 'hardtop', 'hatchback',
                 'notchback', 'ragtop', 'sports',
                 'car', 'sport' 'vehicle', 'wagon', 'SUV', 'compact', 'coupe', 'coupé', 'intermediate',
                 'limousine', 'mini', 'minicar', 'sedan', 'subcompact', 'hybrid', 'Battery',
                 'station', 'station', 'EV', 'Fuel', 'Cell', 'LEVEL', 'Electric', 'Electric',
                 'Vehicle', 'Vehicle', 'Vehicle', 'car', 'car', 'car', 'car', 'automobile',
                 'automobile', 'automobile', 'control', 'driving', 'drive', 'autonomous', 'transport', 'transportation']
    key_words = [x.lower() for x in key_words]
    bosch = req('https://www.bosch-mobility-solutions.com/en/products-and-services/passenger-cars-and-light-commercial-vehicles/powertrain-systems/electric-drive/vehicle-control-unit/');
    ecotron_vcu = req('https://ecotron.ai/vcu-vehicle-control-unit/')

    bosch = [x.lower() for x in bosch]
    ecotron_vcu = [x.lower() for x in ecotron_vcu]

    bosch_copy = bosch.copy()
    ecotron_vcu_copy =  ecotron_vcu.copy()

    for item in bosch:
        if item in string.punctuation or item in prep:
            bosch_copy.remove(item)

    for item in ecotron_vcu:
        if item in string.punctuation or item in prep:
            ecotron_vcu_copy.remove(item)
    key_words = key_words + bosch_copy+ ecotron_vcu_copy

    return key_words

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
        if key in string.punctuation or key in prep:
            dic.pop(key)
        try:
            (int(key))
            dic.pop(key)
        except:
            continue
    return dic

def read_file(load):
    file = pd.ExcelFile('Em_lan.xlsx');
    df = file.parse("Sheet1")
    if not load:
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

    return df


def get_data(load):  # if load==True, load the data; else: run the get_data function

    with open("data.txt", "rb") as fp:
        data = pickle.load(fp)
    if not load:
        for i in tqdm(range(df.__len__())):
            URL = 'https://' + df.company_name[i]
            v_text = req(URL)
            data[i] = v_text
            print(i)
            if i % 1 == 0:
                with open("data.txt", "wb") as fp:
                    pickle.dump(data, fp)
    return data

def req(URL):

    # URL =  'https://'+df.company_name[i]
    url = URL #http

    url_https = "https://" + url.split("//")[1]
    if check_url(url_https):
        # print("Nice, you can load it with https",URL)
        URL = url_https
        time_out = 40
    else:
        if check_url(url):
            print("https didn't load, but you can use http",URL)
            URL = url
            time_out = 40
        else:
            print("both https and http don't work",URL)
            URL = url_https
            time_out = 10
    results = {}
    v_text = []
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36"
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36 Edg/89.0.774.68"
    headers = {"user-agent": USER_AGENT}
    # n = 0
    try:
        resp = urllib.request.Request(URL,data=None, headers=headers)
    except:
        v_text = []
        print("request failed", URL)

    try:
        html = urllib.request.urlopen(resp,timeout=30).read()
        v_text = (text_from_html(html))
        v_text = list(v_text.split(" "))
        v_text = list(filter(None, v_text))
    except:
        v_text = ['TIMEOUT']
        print("request timeout", URL)



            # weights_keywords = get_weights(key_words);
            # weight = get_weights(V_text)
            # n = compare_weights(weights_keywords, weight)


    #Tokenize words and create dictionary

    return  v_text

def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])

def trans(lst):
    if lst!=[] or lst != ["TIMEOUT"]:
        try:
            lan = detect(' '.join(lst))
            print(lan)
            # text = TextBlob(' '.join(lst))
            # lan = text.detect_language()
        except:
            print('failed')
            lan = 'Failed'
    else:
        lan = 'Nan'

    return lan

def update_trans_dic(data):
    for i in tqdm(range(len(data))):

        lan = df.iloc[i]['language']
        if (lan != 'en' and lan != 'timeout' and lan != 'Nan' and lan != 'nan'):
            time.sleep(1.9)
            res = translator.translate(' '.join(data[i]))
            res = re.sub(r'[^\w\s]', '', res)
            res = list(res.split(" "))
            res = list(filter(None, res))
            data[i] = res
            print(lan)
            if i % 100 == 0:
                pickle.dump(data, open("data1.p", "wb"))
    return data





# URL = "https://ecotron.ai/"
# # URL = 'http://' + df.company_name[i]
# n = req(URL)

# df = pd.read_pickle("a_file.pkl")

#

# for i in tqdm(range(2726,len)):
#     URL = "https://ecotron.ai/"
#     URL = 'https://' + df.company_name[i]

    # n = req(URL)
    # df.at[i, "KeyWord Similarity score"] = n
    # # if i %2 == 0:
    # df.to_pickle("a_file.pkl")



# for i in range(df_all.__len__()):
#     data = df.loc[df['company_name'] == df_all.at[i,"company_name"]]
#     data = data['KeyWord Similarity score']
#     if data.empty:
#         df_all.at[i, "KeyWord Similarity score"] = -1
#         print(i)
#     else:
#         df_all.at[i,"KeyWord Similarity score"] = float(data)



if __name__ == '__main__':
    print("being process")
    translator = google_translator()

    prep = (
        "When", 'the', "or", 'introduction', "further", "is", "claim", "wherein", "that", "keep", "causes", "a", "and",
        "are", "be", "an", "aboard",
        "about", "above", "across", "after", "against", "along", "amid", "among", "anti", "around", "as", "at",
        "before",
        "behind", "below", "beneath", "beside", "besides", "between", "beyond", "but", "by", "concerning", "considering"
        , "despite", "down", "during", "except", "excepting", "excluding", "following", "for", "from", "in", "inside",
        "into",
        "like", "minus", "near", "of", "off", "on", "onto", "opposite", "outside", "over", "past", "per", "plus",
        "regarding",
        "round", "save", "since"
        , "than", "through", "to", "toward", "towards", "under", "underneath", "unlike", "until", "up", "upon",
        "versus", "via",
        "with", "within", "without", 'Bosch', 'Group', 'English', 'Deutsch', 'Solutions', 'Highlights', 'Personalized',
        'service', 'Convenience', 'Perfectly',
        '-', 'Think', 'Act', 'En', 'Projects', 'initiatives', 'Connected', 'Services', 'Updates', 'over', 'agriculture',
        'your', 'mix', 'quality', 'Breakthrough', 'quality',
        'business', 'air', 'Perfectly', 'Connected', 'services', 'HMI', 'solutions', 'Community-based', 'parking',
        'Powertrain', 'and', 'electrified', 'mobility',
        'Electromobility', 'The', 'future', 'of', 'diesel', 'Urban', 'mobility', 'and', 'air', 'quality', 'Powertrain',
        'mix', 'for', 'better', 'air', 'quality',
        '©', '2021', 'Robert', 'Bosch', 'GmbH.', 'All', 'rights', 'reserved.', 'Company', 'information', 'Legal',
        'notice', 'Data', 'protection', 'notice', 'Privacy',
        'settings', 'EAXVA04', 'CES', '2020', 'MathWorks', 'Partner', 'Helpful', 'Link', '©', '2021', 'Ecotron.', 'All',
        'Rights', 'Reserved.', '|', 'Videezy',
        'HOME', 'PRODUCTS', 'ADCU', 'ADAS', 'EcoCoder–AI', 'EV', 'Products', 'VCU', 'SCU', 'TCU', 'EcoCoder', 'EcoCAL',
        'EcoFlash', 'NEWS', 'DOWNLOADS', 'CAREERS', 'CONTACT')

    prep = [x.lower() for x in prep]

    key_words = key_words_gen()
    weights_keywords = get_weights(key_words);


    df = read_file(True)
    del df['Unnamed: 0']
    df = df.replace('vi', 'timeout')


    df.index = range(df.__len__())

    # data_copy = get_data(True)
    data = pickle.load(open("data1.p", "rb"))

    i = 0
    for i in tqdm(range(data.__len__())):
        n = -1
        v_text = data[i]
        if v_text == ['TIMEOUT']:
            n = -1
        else:
            weight = get_weights(v_text)
            n = round(compare_weights(weights_keywords, weight))/10
        df.at[i,"relevance_socre"] = n

    df.to_excel("Em_result.xlsx")



    # for i in tqdm(range(len(data))):
    #
    #     lan = trans(data[i])
    #     df.at[i, 'language'] = lan




    # with open("key_words.txt", "wb") as fp:
    #     pickle.dump(key_words, fp)
    # with open("key_words.txt", "rb") as fp:
    #     key_words = pickle.load(fp)
    # remove duplicates
    # df.drop_duplicates(subset ="company_name",
    #                      keep = 'first', inplace = True)


