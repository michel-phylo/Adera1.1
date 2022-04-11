

#Adera is a question-answer drug repurposing software. Its main workflow includes
#taking the two search inputs. The first is the pathways or diseases needed to repurpose from.
#the other one is the pathway or disease that needed to be repurposed to.
#The software then downloads the PUBMED ID INDEX for the first query (e.g., repurposed from). Then it extracts the keywords from it and 
#only downloads the terms that have does not have a semantic relationship with the second query( repurposed to).
#then the software downloads the relevant Pdfs based on the results of the last step.
#The program then uses two AI networks. the first calculates the embedding (numerical representation of each sentence in the downloaded Pdfs.
# The second network calculates the relevance.
#the output of the software is a table showing sorted answers from one pdfs of or multiple pdfs
#In the upcoming release the software will include the GUI direct interface and PyPI




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
#pip3 install bs4
import os
#os.system('pip3 install metapub')



import os



# Check whether the specified path exists or not
isExist = os.path.exists("fetched_pdfs")

if not isExist:
  
  # Create a new directory because it does not exist 
  os.makedirs("fetched_pdfs")
  print("fetched_pdfs created..Thank God")
#os.makedirs("fetched_pdfs")


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

#print(kkr)
print("adera2")

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


#print(mkr5)
print("adera3")
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
os.chdir('fetched_pdfs')

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
nltk.download('punkt')
#import textract
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#pip install tika 
from tika import parser
#pip install tika-app
import tika




path_adera= 'fetched_pdfs'
kyrillos_name1=os.listdir()[0]
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

path_adera=os.getcwd()
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
for n in range(0,6):#,len(datastore)):
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

kr8=kr3.sort() # this is some numerical values represneting cosine distance
#print(kr3) # this is a column of numbers
#kr5b = kr3.index(kr3[-1])
#print('kr5b',kr5b)
#datastore[3]['data'][kr5b]

#for j in range(1,10):
  # kr5b = kr8.index(kr3[-j])
   #print(kr5b)
  # print(datastore[5]['data'][kr5b]) 
# the number in bracktes is the number  of the pdf being checked

from sh import cd, ls

cd ("..")
with open('3ad1relevance_test_database.json','r') as f:     
    dataspore = json.load(f)
    print(dataspore)
    dataspore['data'][5]
    b5=dataspore['data']
    kessages=b5
print(kessages)



with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        kessage_embeddings = session.run(embed(kessages))
        print(kessages)
        print(kessage_embeddings.shape)
        print(kessage_embeddings)


print(kessages [0])
print(kessage_embeddings.shape)


#this function calculates similairty 
def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err



#second solution
from sklearn.model_selection import train_test_split
x_mkr=kessage_embeddings[0::2]
y_mkr=kessage_embeddings[1::2]
X_train,X_test,y_train,y_test=train_test_split(x_mkr,y_mkr,test_size=0.2)
X_test
X_train=X_train.reshape(3,128,4,1)
y_train=y_train.reshape(3,128,4,1)
X_test=X_test.reshape(1,128,4,1)
y_test=y_test.reshape(1,128,4,1)
from keras.models import Sequential #need to know how to add more layers
from keras.layers import Dense, Conv2D, Conv3D,Flatten,Conv1D
#create model
model = Sequential()
#add model layers
#model.add.Conv1D(2#, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, )
model.add(Conv2D(2, (1, 1), activation='relu', input_shape=( 128, 4, 1)))
#model.add(Conv3D(1,kernel_size=3,Input(batch_shape=( 128, 4, 1))))
#model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train,epochs=5,verbose=1,validation_data=(X_test, y_test))


z0=message_embeddings[0].reshape(1,128,4,1)
z0a=model.predict(z0)


results = []
#for n in range(0,len(datastore['data'])):
for n in range(0,4):
#z1=np.random.randint(1, size=(2, 4,1))
  z1=message_embeddings[n].reshape(1,128,4,1)
  z1mkr=model.predict(z1)
  m1 = mse(z0, z1mkr)
  #print("this is data in datastore",n)
  #print(datastore [n]['data']) # this is cobined data from the question and a pdf number 3
  #print(m1)
  #results.append(datastore['data'][n])
  results.append(m1)
#print("results",results)
np.hstack(results)
def merge(list1, list2): 
      
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))] 
    return merged_list 
#c2=merge(results,datastore['data'])
c2=merge(results,datastore[3]['data'])

c2.sort(reverse=False)
#print(c2)
#c2
import pandas

# Creating a dataframe object from listoftuples
dfObj = pd.DataFrame(c2)
print(dfObj)
