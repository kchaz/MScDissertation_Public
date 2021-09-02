# -*- coding: utf-8 -*-
"""

This file load, pre-processes, and explores the Demography data

Only runs pre-processing steps if cannot load the files from data directory

Author: Kyla Chasalow
#import os
Last edited: July 23, 2021
"""
#standard imports
import sys
import os
import pandas as pd
import numpy as np
import pickle


# Import relative paths
from filepaths import code_path
from filepaths import demog_data_path as data_path
from filepaths import demog_eda_plots  as plot_path

#import other modules
sys.path.insert(0, code_path)
from Helpers import file_override_request
import AbstractCleaner

#saving settings
override = file_override_request()
dpi = 200


#try loading final data
try:
    TM_data = pd.read_csv(os.path.join(data_path,'Demog_data_final.csv'))
    print("Dataset shape:", TM_data.shape)
except:
     message =  "Demog_data_final.csv does not exist. This file is required to run this script"
     raise Exception(message)
   
abstracts = TM_data["dc:description"]
     

#remove "Population Association of America" instances - appears in vast majority of abstracts because
#of copyright  

count1 = 0
count2 = 0
count3 = 0
inds = []
for i,a in enumerate(abstracts):
    if "Population Association of America" in a:
        count1+=1
        inds.append(i)
    if "population association" in a:
        count2 +=1
    if "Population Association" in a:
        count3 +=1

print("%d abstracts contain 'Population Association of America'" % count1)
print("%d contain 'population association' (lowercase)" % (count2) )
print("%d contain only 'Population Association' without `of America'" % (count3-count1) )
print("Removing 'Population Association of America' instances")

#abstracts is now a list of strings
abstracts = AbstractCleaner.remove_phrase(list(abstracts), remove_list = ["Population Association of America"])


     
#try loading first step of cleaned abstracts or generate them if not present
try:
    with open(os.path.join(data_path,"Demog_abstracts1_Basic.pickle"), "rb") as fp:   #Unpickling
        cleaned1 = pickle.load(fp)
    print("loaded process 1 (basic cleaning) abstracts from file")
except: 
    print("running process 1: basics")
    stoplist = AbstractCleaner.create_stop_list()
    cleaned1 = AbstractCleaner.basic_cleaner(abstract_list = abstracts,
                                             stoplist = stoplist,
                                             save = True,
                                             override = override,
                                             filename = "Demog_abstracts1_Basic",
                                             outpath = data_path)

print("----------------------------")
print("After processes 0 and 1:")
vocab1 = AbstractCleaner.extract_vocab(cleaned1)
vocab1_len = len(vocab1)
corpus_len = AbstractCleaner.corpus_length(cleaned1)
print(len(vocab1), "words in vocab")
print(corpus_len, "words in corpus")



#second step
print("----------------------------")
try:
    with open(os.path.join(data_path,"Demog_abstracts2_Ly.pickle"), "rb") as fp:   #Unpickling
        cleaned2 = pickle.load(fp)
    print("loaded process 2 (ly words) abstracts from file")
except:
    print("running process 2: ly words")
    cleaned2 = AbstractCleaner.fix_adverbs(cleaned1, 
                                           save = True,
                                           override = override,
                                           filename = "Demog_abstracts2_Ly",
                                           outpath = data_path)

print("After process 2:")
print(len(AbstractCleaner.extract_vocab(cleaned2)), "words in vocab")
print(AbstractCleaner.corpus_length(cleaned2), "words in corpus")



#third step
print("----------------------------")
cutoff = 25
print("Bigram cut-off: require %d or more occurences" % cutoff)


try:
    with open(os.path.join(data_path,"Demog_abstracts3_BigramAdder.pickle"), "rb") as fp:   #Unpickling
        cleaned3 = pickle.load(fp)
        
    with open(os.path.join(data_path,"Bigram_list.pickle"), "rb") as fp:   #Unpickling
        bigram_list = pickle.load(fp)
    print("Loaded process 3 (bigram adder) abstracts from file and loaded list of bigrams")
        
except:
    print("running process 3: bigram adder (second bigram option)")
    cleaned3, bigram_list = AbstractCleaner.bigram_adder(cleaned2, 
                                           thresh = cutoff,
                                           save_docs = True,
                                           override = override,
                                           doc_filename = "Demog_abstracts3_BigramAdder",
                                           doc_outpath = data_path,
                                           save_bigrams = True,
                                           bigram_filename = "Bigram_list",
                                           bigram_outpath = data_path)

print("After process 3:")
print(len(AbstractCleaner.extract_vocab(cleaned3)), "words in vocab")
print(AbstractCleaner.corpus_length(cleaned3), "words in corpus")

print("-------------------------------------")
print("%d BIGRAMS" % len(bigram_list))
print(bigram_list)
print("-------------------------------------")




# apply last pre-processing step (removing words in under 5 documents)
try:
    with open(os.path.join(data_path,"Demog_abstracts4_FreqFilter.pickle"), "rb") as fp:   #Unpickling
        cleaned4 = pickle.load(fp)
    print("Loaded process 4 (frequency filter, remove words in <5 docs) abstracts from file")
except:  
    print("Removing words appearing in under 5 documents:")
    cleaned4 = AbstractCleaner.filter_by_word_doc_freq(cleaned3,
                                                       no_below = 5,
                                                       no_above = 1,
                                                       save_docs = True,
                                                       override = override,
                                                       doc_filename = "Demog_abstracts4_FreqFilter",
                                                       doc_outpath = data_path)

print("After process 4:")
print(len(AbstractCleaner.extract_vocab(cleaned4)), "words in vocab")
print(AbstractCleaner.corpus_length(cleaned4), "words in corpus")




print("\n\n-------------------------------------")
print("Getting dictionary and corpus")
corpus, dictionary = AbstractCleaner.get_corpus_and_dictionary(cleaned4)
print("-------------------------------------")


print("\n 20 randomly selected example words that were removed:")
#using fact that words that have frequency < 5 will in any case appear in < 5 documents...
#rare_words is not per se a complete list of all words removed - just allows me to take a
#quick easy peek
fdist3 = AbstractCleaner.corpus_FreqDist(cleaned3)
rare_words = [word for word in fdist3.keys() if fdist3[word] < 5]
inds = np.random.randint(low = 0, high = len(rare_words), size = 20)
print([rare_words[i] for i in inds])


print("\n Bigrams removed:")
words_list = dictionary.token2id.keys()
removed = []
for word in bigram_list:
    if word not in words_list:
        removed.append(word)
print(removed)



print("-------------------------------------")
print("-------------------------------------")
print("Generating plot of final abstract lengths and document counts")



lengths = AbstractCleaner.abstract_wordcounts(cleaned4, 
                                                    years = TM_data.Year,
                                                    plot = True,
                                                    color = "purple", alpha = 0.45,
                                                    save_fig = True,
                                                    fig_override = override,
                                                    fig_name = "abstract_length_grid",
                                                    fig_outpath = plot_path,
                                                    dpi = dpi
                                                   )



print("-------------------------------------")
print("\n A look at some of the really short documents (<20 words long after pre-processing) \n")
under20 = []
under20_ind = []
over200 = 0

for i, elem in enumerate(cleaned4):
    if len(elem) < 20:
        under20.append(elem)
        under20_ind.append(i) 
        print(abstracts[i])
        print(cleaned4[i])
        print("\n")
    elif len(elem) > 200:
        over200 += 1
        
        
    

print("There are %d documents with length < 20" % len(under20))
print("There are %d documents with length > 200" % over200)
        
print("-------------------------------------")
print("Done!")