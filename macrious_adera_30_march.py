

Adera program, this program aims to take the search input from the user and downalod the PUBMED id INDEX FOR THEM. Then it extrcats the keywords from it and  only downlaods the terms that ahve got one of the key words but not the other.
The second phases of the search will be done in the sesond part which includes parsing the files and generating compound names.
The third phase coming month is building a wrapper that the user can inetrcat with.
"""



import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict


#pip install pubmed2pdf
#pip install keybert
#pip install textract
#pip install nltk
#pip install fstrings 
#pip install tika 
#pip install tika-app
#pip3 install --quiet "tensorflow>=1.7"
# Install TF-Hub.
#pip3 install --quiet tensorflow-hub
#pip3 install --quiet seaborn
#pip3 install --quiet chainer
import os
#os.system('pip install metapub')


keyword_macrious= input("please enter a search term :  ")
exclude_macroius= input ("please enter a disease to purpose to :" )
name_of_json_adera=input("please enter a name for your database. The name has to be 'mkr_data_11_feba.json")


from metapub import PubMedFetcher
fetch = PubMedFetcher()
print("adera1")

pmids_adera = fetch.pmids_for_query(keyword_macrious, pmc_only= True,retmax=20)


# get abstract for each article:
abstracts_adera = {}
for pmid_adera in pmids_adera:
    abstracts_adera[pmid_adera] = fetch.article_by_pmid(pmid_adera).abstract
print(pmids_adera)
#print(abstracts_adera)
#mkr1 = list(abstracts_adera.values())
mkr1= abstracts_adera.keys()
#print(mkr1)
mkr2=abstracts_adera.values()
#print(mkr2)
mkr3=list(abstracts_adera)
#mkr3[0]
mkr4=mkr3

from keybert import KeyBERT
kw_model = KeyBERT()
kkr=[]
for x in range(1, len(abstracts_adera)):
  #print(x)  
  keywords_adera = kw_model.extract_keywords(abstracts_adera[mkr3[x]])
  print('keywords_adera  ', keywords_adera )
  #for ii in range(0,4):
      #print(keywords_adera[ii][0])
      #if keywords_adera[ii][0]==exclude_macroius:
        #print("adera4")
      #else:
        #kkr.append(keywords_adera)

print(kkr)

##3until here the output is the keywords of each abstract
import numpy as np
result_adera=[]
None in kkr

for x in range(0, len(kkr)):
    if kkr[x]== None:
        print(x)
        result_adera.append(x)

print(result_adera)

result_adera1=np.array(result_adera)
result2_adera=result_adera1.ravel()

if not result2_adera:
   print ("macrious1")
   mkr5=mkr4
else:
    mkr5=np.delete(mkr4, result2_adera).tolist()


print(mkr5)

mkr5d= ",".join([str(elem) for elem in mkr5])
print("mkr5d =", mkr5d)
text_file_adera="adera4_results.txt"
text_file = open(text_file_adera, "w")

text_file.write(mkr5d)
text_file.close()   
print("macrious 2")


from fstrings import f
print("mkr5d",mkr5d)
import os
os.system(f'python3 fetch_pdfs.py -pmids {mkr5d}')
os.chdir('/fetched_pdfs')

arr_adera = os.listdir()
print("arr_adera =", arr_adera)

os.system('find . -type f ! -iname "*.pdf" -delete')
#!find . -type f ! -iname "*.pdf" -delete
kr7=os.path.isdir('default')
print(kr7)
if kr7==True :
   os.rmdir ('default')

#!ls

print("now we have downaloded the pdf and cleaned up_adera")
###step of converting pdfs to json
import gc
gc.collect()

import nltk
#nltk.download('punkt')
#import textract
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#pip install tika 
from tika import parser
#pip install tika-app
import tika




path_adera= '/fetched_pdfs'
kyrillos_name1=os.listdir(path_adera)[0]
new_file_name=kyrillos_name1
raw1= parser.from_file(new_file_name)
import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

print("adera4")


data1 = raw1['content']
from nltk import sent_tokenize
#print (sent_tokenize(data1))#
b1=sent_tokenize(data1)
#print(b1)
print("now we have tokenized_adera2")
import pandas 
#need to add a question for the user to choose a name for the json folder

############# write the json database 
file_folder_mkr= name_of_json_adera
import json

# Make it work for Python 2+3 and with Unicode
import io
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

# Define data
data = {'threasa': 1,
        'name': new_file_name,
        'data': b1}

# Write JSON file
with io.open(file_folder_mkr, 'w', encoding='utf8') as outfile:
    str_ = json.dumps(data,
                      indent=4, sort_keys=True,
                      separators=(',', ': '), ensure_ascii=False)
    outfile.write(to_unicode(str_))
#####################################################

import json

# Make it work for Python 2+3 and with Unicode
import io
try:
    to_unicode = unicode
except NameError:
    to_unicode = str
   
    
#for loop add all files in folder
import os
import nltk.data
from nltk import sent_tokenize
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
Adera = os.listdir()

with open(file_folder_mkr, "r") as read_file:
     brata = json.load(read_file)
grata = [brata]
#print(grata)
#Adera[1]
import os

onlyfiles = next(os.walk(path_adera))[2] #dir is your directory path as string
kk=len(onlyfiles)
for i in range(1,kk): 
    new_file_name=str(Adera[i])
    raw1= parser.from_file(new_file_name)
    data1 = raw1['content']
    data1 = str(data1) 
   # safe_text = data1.encode('utf-8', errors='ignore')
    krm1=sent_tokenize(data1)
    a_dict = {'data': krm1,'threasa':i,'name':new_file_name }
   # print(a_dict)
    grata.append(a_dict)
#print(grata)

with open(file_folder_mkr, 'w') as f:
    json.dump(grata, f)
print("now the first phase has conclude, thanks be to God")


import gc 
gc.collect()

import json


import pandas 
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
import numpy as np
from chainer import Variable, optimizers
import matplotlib.pyplot as plt
import nltk

nltk.download('punkt')
from keras import backend
from keras.layers import Activation, Dense, Input, Subtract
from keras.models import Model

from absl import logging

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
model = hub.load(module_url)
print ("module %s loaded" % module_url)
def embed(input):
    return model(input)


with tf.Session() as session:
     session.run([tf.global_variables_initializer(), tf.tables_initializer()])

from scipy.spatial.distance import directed_hausdorff
import numpy as np
def plot_similarity(labels, features, rotation):
    corr = np.inner(features, features)
    #print(type(features))
    l=sorted(corr[0])[-2] #   takes the highest value only in the first row ( the row of the question)
    ll = np.where(corr[0] == l)
    o=messages[ll[0][0]]
    print('mkr____________',o)

def run_and_plot(session_, input_tensor_, messages_, encoding_tensor):
    message_embeddings_ = session_.run(
       encoding_tensor, feed_dict={input_tensor_: messages_})
    plot_similarity(messages_, message_embeddings_, 90)

import json
sentence=keyword_macrious
with open(name_of_json_adera,'r') as f:    
          datastore = json.load(f)
for n in range(5,6):#,len(datastore)):
    print(n)
    paragraph=datastore[n]['data']
    a5=paragraph
   # print(a5)
    print("a555555555555")
    a5.insert(0,sentence)
    messages=a5
    print("adera_messages")
   # print(messages)
    with tf.Session() as session:
         session.run([tf.global_variables_initializer(), tf.tables_initializer()])
         message_embeddings = session.run(embed(messages))
#print('messages',messages)
print(message_embeddings.shape)

print(message_embeddings)
import matplotlib.pyplot as plt
import numpy as np
#sns.heatmap(message_embeddings,xticklabels=5, yticklabels=True)
#plt.imshow(message_embeddings)
#plt.show()

kr3=[]
from scipy.spatial import distance
kr1=message_embeddings[1,] #why [1,] ?
#print('kr1',kr1)

for i in range(1,len(message_embeddings)):
    kr2=message_embeddings[i,]
    kr4=1 - distance.cosine (kr1,kr2)
    kr3.append(kr4)

kr3.sort() # this is some numerical values represneting cosine distance
#print(kr3) # this is a column of numbers
#kr5b = kr3.index(kr3[-1])
#print('kr5b',kr5b)
#datastore[3]['data'][kr5b]

for j in range(1,10):
   kr5b = kr3.index(kr3[-j])
   print(kr5b)
   print(datastore[5]['data'][kr5b]) # the number in bracktes is the number  of the pdf being cheked