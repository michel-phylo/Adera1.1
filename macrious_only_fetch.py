#from fstrings import f
#print("mkr5d",mkr5d)

import os
#os.system(f'python3 -m pubmed2pdf pdf --pmids {entry1}')
os.system(f'python3 fetch_pdfs.py -pmids  {entry1}')

#for i in(0,len(mkr5d)-1):
     #os.system(f'python3 -m pubmed2pdf pdf --pmids {entry1}')
#     print({mkr5d[i]})

#for i in(0,len(mkr5d)-1):
    #os.system(f'python3 fetch_pdfs.py -pmids {mkr5d[i]}')
    #print({mkr5d[i]})


#os.chdir('fetched_pdfs')

arr_adera = os.listdir()
print("arr_adera =", arr_adera)

