# -*- coding: utf-8 -*-
"""
This script sets up an interactive prompt for exploring the top documents from a given topic and model
It also will display the top words for selected topic



Author: Kyla Chasalow
Last edited: August 31, 2021
"""


#standard imports
import sys
import pickle
import os
import logging
import numpy as np
import pandas as pd


#gensim imports
from gensim.models.ldamodel import LdaModel

# Import relative paths
from filepaths import code_path
from filepaths import demog_data_path as data_path
from filepaths import demog_models_path as model_path
from filepaths import demog_models_matrix_path as matrix_path

# Import functions
sys.path.insert(0, code_path)#r'C:\Users\kcsky\Documents\Oxford\STATISTICS\Dissertation\GitHub\MScDissertation\src')
import AbstractCleaner
import LdaOutputDocs
import LdaOutput




#set dpi for saving figures
dpi = 150


#Load data and extract year labels
TM_data_final = pd.read_csv(os.path.join(data_path,"Demog_data_final.csv"))
year_labels = TM_data_final.Year 

# Load processed abstracts and get corpus and dictionary
filepath = os.path.join(data_path, "Demog_abstracts4_FreqFilter.pickle")
with open(filepath, "rb") as fp:   #Unpickling
    cleaned_docs = pickle.load(fp)
print(len(AbstractCleaner.extract_vocab(cleaned_docs)), "words in vocab")
print(AbstractCleaner.corpus_length(cleaned_docs), "words in corpus")
corpus, dictionary = AbstractCleaner.get_corpus_and_dictionary(cleaned_docs)




# SET UP LOGGING TO CONSOLE
#create instance completely separate from other log
logger = logging.getLogger() 
#create handler and set its properties
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s : %(levelname)s - %(message)s")
handler.setFormatter(formatter)
#add handler to logger
logger.addHandler(handler) 






#LOAD ALL THE MODELS
Kvals = [5, 10, 15, 20, 25, 30 ,35, 40, 50, 60]


# LOAD ALL THE MODELS
fnames = ["Best_%d_model"%k for k in Kvals]
model_dict = {}
for i,k in enumerate(Kvals):
    model_dict[k] = LdaModel.load(os.path.join(model_path,fnames[i]))
    


      
# Load saved theta matrices into dictionary
fnames = ["theta_matrix_" + str(k) + ".npy" for k in Kvals]
theta_dict = {}
for i, k in enumerate(Kvals):
    theta_dict[k] = np.load(os.path.join(matrix_path, fnames[i]))


print("\n\n\n\n\n")

################################################################

def global_prompt():
    print("\nGlobal settings: Choose output mode and model. \n")
    print("Note: to change these, quit iteractive prompt and start over") #make it so can restart
    
     
    #give latex version or not?
    message = "OUTPUT MODE: Output Latex version? (Yes/No) (if no, pandas dataframe)  "
    answer = str(input(message))
    while answer.lower() not in ["yes","no"]:
        print("Invalid value, try again")
        answer = input(message) 
    if answer.lower() == "yes":
        latex = True
    else:
        latex = False 
        
    #model to look at
    message = "MODEL: Select K-topic model: choices are %s  " % str(Kvals)
    k = int(input(message))
    while k not in Kvals:
        print("Invalid value: try again")
        k = int(input(message))
  
    #how many top words to show
    message = "WORDS: How many top words should I show when displaying topic words?"
    topn = int(input(message))
    while topn < 0 or topn > len(AbstractCleaner.extract_vocab(cleaned_docs)):
        print("Invalid Value: try again")
        topn = int(input(message))
    
    return(latex, k, topn)




def local_prompt():
    
    #topic number 
    message = "Select topic number in range 0-%d  " % (k-1)
    i = int(input(message))
    while i < 0 or i >= k:
        print("Invalid value, try again")
        i = int(input(message))
    
    
    #number of top documents to show
    num_doc = TM_data_final.shape[0]
    message = "How many top documents would you like to see? (enter integer in (0,%d)  " % num_doc
    topd = int(input(message))
    while topd < 0 or topd > num_doc:
        print("Invalid value, try again")
        topd = int(input(message))
    
    #include abstracts or not
    message = "Show abstracts? (Yes/No)  "
    answer = str(input(message))
    while answer.lower() not in ["yes","no"]:
        print("Invalid value, try again")
        answer = input(message) 
    if answer.lower() == "yes":
        omit_abstract = False
    else:
        omit_abstract = True
    
    return(i, topd, omit_abstract)




def look_at_topic(latex, k, topn):
    """Once global settings set, this handles rest"""
    
    i, topd, omit_abstract = local_prompt()
    
    print("\n \n \n-----------------")
    print("%d-Topic Model:  Topic %d, Top %d documents by theta" % (k,i,topd))
    print("-----------------\n")
    out = LdaOutputDocs.get_topic_topdoc_table(TM_data_final,
                                      theta_dict[k], 
                                      topic_id = i,
                                      topd = topd, 
                                      latex = latex, 
                                      omit_abstract = omit_abstract)

    
    top_words = LdaOutput.get_top_words(model = model_dict[k], 
                                         topic_id = i,
                                         dictionary = dictionary,
                                         topn = topn)
    

    #pd.set_option('display.max_colwidth', None) #setting so that displays entire column
    if latex:
        print("Top %d words:" % topn, top_words, "\n\n")
        print(out)
    else: 
        print("Top %d words: "% topn, top_words, "\n\n" )
        print(out.to_string())
    
    print("\n \n \n ")


   
    
    
#get global settings
latex, k, topn = global_prompt()
print("Global settings set")
print("--------------------\n")

#repeatedly iterate local topic view and settings as long as user desires
look_at_topic(latex, k, topn)

another_topic = True
while another_topic:
    message = "Display another topic? (Yes/No)"
    answer = str(input(message))
    while answer.lower() not in ["yes","no"]:
        print("Invalid answer, try again (Yes/No)")
        answer = input(message) 
    if answer.lower() == "yes":
        another_topic = True
        look_at_topic(latex, k, topn)
    else:
        another_topic = False
        print("quitting interactive prompt")
