# Adera1.1
Adera1.1 is a question-answer drug repurposing software. The primary task of this software is to repurpose relevant known drugs published in the literature for a  particular disease specified by the user. The current workflow of Adera1.1.1 is as follows:\

a)  The search input includes a query question and a disease.  Based on these two inputs, Adera1.1  searches PUBMED for medications. It then downloads the abstract and the PubMed ID a specific number of times. \
 b) Then, it extracts the keywords from each abstract. Subsequently, it downloads full pdfs of the articles relevant to the query question. It also filters out the articles that already cover the disease specified. This step ensures that the purposed medications have not been published as drugs for the particular disease. \
c) After that, using an AI encoder, the software embeds all the sentences in each downloaded pdf together with the query question. Finally, it calculates the relevance between each sentence and the query question.\
d) Sentences with the highest relevance are shown on the screen to the user. \
e) In Adera1.2 a wrapper will be added using python tinker and Docker.\
 
# Installing on ubuntu 
To install Adera1.1, follow the normal procdure to install a git reporsitory. i.e by doing the following :
```
sudo apt-get update
sudo apt-get install git
git clone https://github.com/michel-phylo/Adera1.1
cd Adera1.1
python3 adera1_req.py
python3 adera_tinker_19_may.py
```

# Installing on macOS
The easiest way is  to install the Xcode Command Line Tools. On Mavericks (10.9) or above you can do this  by trying to run git from the Terminal the very first time abd then follow the interactive dialogue.
```
git --version
git clone https://github.com/michel-phylo/Adera1.1
cd Adera1.1
python3 adera1_req.py
python3 adera_tinker_19_may.py
```

# Installing for windows
First you would need to install git on windows.You can try https://git-scm.com/downloads and follow the instructions. After that, you just clone the git repo and run it as follows
```
git --version
git clone https://github.com/michel-phylo/Adera1.1
python3 adera1_req.py
python3 macrious_adera_2_april.py
```


# Tutorial
The software interactive mode is pretty straight forward. \
First you need to run the command 
```
python3 adera_tinker_19_may.py
```
An interactive dialogue will appear. 
please enter a search term: what are the drugs used to regulate Nrf2 function\
please enter a disease to purpose to :brain \
please enter a name for your database:mkr1.json\

# Exe-based program
To make the program accessible for usage, we created two exe options\
(i) option one, the program was converted to exe-based software, so people do not need to learn python or coding to use it. However, the program is quite large because of tensor flow. We will upload it here shortly.\
(ii) The second option the software workflow is divided into four exe-based programs\
a) The first program could be used to search PubMed and other science-based websites for PDFs that could carry the answer to the user's question\
b) The second program is concerned with fetching the PDF. It could be used alone or in conjunction with the previous step\
c) The third program is concerned with parsing the downloaded pdf and converting them into a JSON-based database.\
d) the fourth program is concerned with calculating the final results and presenting the answers to the user.\
We will upload the programs to different branches, so they are avaiubale for download by interested users.\


