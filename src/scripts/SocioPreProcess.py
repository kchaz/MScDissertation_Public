# -*- coding: utf-8 -*-
"""

This file load, pre-processes, and explores the Sociology data

Only runs pre-processing steps if cannot load the files from data directory

Author: Kyla Chasalow

Last edited: August 10, 2021
"""
#standard imports
import sys
import os
import pandas as pd
import numpy as np
import pickle
from pprint import pprint

# Import relative paths
from filepaths import code_path
from filepaths import socio_data_path as data_path
from filepaths import socio_eda_plots  as plot_path

#import other modules
sys.path.insert(0, code_path)
from Helpers import file_override_request
import AbstractCleaner


#saving settings
override = file_override_request()
dpi = 200


#try loading final data
try:
    TM_data = pd.read_csv(os.path.join(data_path,'Socio_data_final.csv'))
    print("Dataset shape:", TM_data.shape)
except:
     message =  "Socio_data_final.csv does not exist. This file is required to run this script"
     raise Exception(message)
   
    
abstracts = list(TM_data["dc:description"])


#-------------------------------------------------------
# STEP 1: Remove Copyright
#-------------------------------------------------------
print("\n Number of © originally present in abstracts")
count, ind, abst = AbstractCleaner.count_copyright_symbols(abstracts)
print(count)


print("\n removing copyright statements")
abstracts, remove_list, not_removed = AbstractCleaner.copyright_remover(abstracts,
                                                                        apply_manual_filter = False)

print("\n Number of © present in abstracts after removal")
count, ind, abst = AbstractCleaner.count_copyright_symbols(abstracts)
print(count)
print("Abstracts containing ©:")
for a in abst:
    print("\n\n\n", a)


print("\n \n \n Phrases removed")
org_list = AbstractCleaner.summarize_copyright_list(remove_list)
pprint(org_list)

# print("\n Detected phrases containing © but not removed because of manual filter:")
# print(not_removed)
     


    
#try loading first step of cleaned abstracts or generate them if not present
try:
    with open(os.path.join(data_path,"Socio_abstracts1_Basic.pickle"), "rb") as fp:   #Unpickling
        cleaned1 = pickle.load(fp)
        print("loaded process 1 (basic cleaning) abstracts from file")
except: 
    print("running process 1: basics")
    stoplist = AbstractCleaner.create_stop_list()
    cleaned1 = AbstractCleaner.basic_cleaner(abstract_list = abstracts,
                                              stoplist = stoplist,
                                              save = True,
                                              override = override,
                                              filename = "Socio_abstracts1_Basic",
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
    with open(os.path.join(data_path,"Socio_abstracts2_Ly.pickle"), "rb") as fp:   #Unpickling
        cleaned2 = pickle.load(fp)
        print("loaded process 2 (ly words) abstracts from file")
except:
    print("running process 2: ly words")
    cleaned2 = AbstractCleaner.fix_adverbs(cleaned1, 
                                            save = True,
                                            override = override,
                                            filename = "Socio_abstracts2_Ly",
                                            outpath = data_path)

print("After process 2:")
vocab2 = AbstractCleaner.extract_vocab(cleaned2)
vocab2_len = len(vocab2)
corpus_len2 = AbstractCleaner.corpus_length(cleaned2)
print(len(vocab2), "words in vocab")
print(corpus_len2, "words in corpus")



#third step
print("----------------------------")
cutoff = 25
print("Bigram cut-off: require %d or more occurences" % cutoff)
print("----------------------------")
try:
    with open(os.path.join(data_path,"Socio_abstracts3_BigramAdder.pickle"), "rb") as fp:   #Unpickling
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
                                            doc_filename = "Socio_abstracts3_BigramAdder",
                                            doc_outpath = data_path,
                                            save_bigrams = True,
                                            bigram_filename = "Bigram_list",
                                            bigram_outpath = data_path)

print("After process 3: Bigram Adder option")
print(len(AbstractCleaner.extract_vocab(cleaned3)), "words in vocab")
print(AbstractCleaner.corpus_length(cleaned3), "words in corpus")

print("-------------------------------------")
print("%d BIGRAMS" % len(bigram_list))
print(bigram_list)


print("\n \n Vocabulary size and overlap by journal after steps 0-3")
journals = TM_data["prism:publicationName"]
out_dict = AbstractCleaner.get_vocab_summary_by_group(cleaned3, journals)
pprint(out_dict["lengths"])
pprint(out_dict["overlap_matrix"])


print("-------------------------------------")

try:
    with open(os.path.join(data_path,"Socio_abstracts4_FreqFilter.pickle"), "rb") as fp:   #Unpickling
        cleaned4 = pickle.load(fp)
    print("Loaded process 4 (frequency filter, remove words in <5 docs) abstracts from file")
except:  
    print("Process 4: Removing words appearing in under 5 documents:")
    cleaned4 = AbstractCleaner.filter_by_word_doc_freq(cleaned3,
                                                       no_below = 5,
                                                       no_above = 1,
                                                       save_docs = True,
                                                       override = override,
                                                       doc_filename = "Socio_abstracts4_FreqFilter",
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
print("Vocabulary Breakdown by group")

journals = TM_data["prism:publicationName"]
print("Corpus lengths by journal")
pprint(AbstractCleaner.get_corpus_size_by_group(cleaned4, journals))
print("Vocabulary size and overlap by journal")
pprint(AbstractCleaner.get_vocab_summary_by_group(cleaned4, journals))



print("-------------------------------------")
print("-------------------------------------")
print("Generating plot of abstract lengths for each journal")

journals = TM_data["prism:publicationName"]
years = TM_data.Year
lengths = np.array([len(elem) for elem in cleaned4])

#word count plot
AbstractCleaner.abstract_length_by_journal(journals, years, lengths, 
                                             length_type = "word",
                                             color_list = ["blue","green","purple"],
                                             vert = True,
                                             figsize = (18,7),
                                             xticks_rotation = 90,
                                             save_fig = True,
                                             fig_name = "socio_wordcounts_by_journal",
                                             dpi = dpi,
                                             fig_override = override, 
                                             fig_outpath = plot_path)


#character count plot
lengths = np.array([len(elem) for elem in TM_data["dc:description"]])
AbstractCleaner.abstract_length_by_journal(journals, years, lengths,
                                            length_type = "character",
                                             color_list = ["blue","green","purple"],
                                             xticks_rotation = 90,                                            
                                             save_fig = True,
                                             fig_name = "socio_charcounts_by_journal",
                                             dpi = dpi,
                                             fig_override = override, 
                                             fig_outpath = plot_path)





print("-------------------------------------")
print("\n A look at the really short documents (<25 words long after pre-processing) \n")
under25 = []
under25_ind = []
over200 = 0

for i, elem in enumerate(cleaned4):
    if len(elem) < 20:
        under25.append(elem)
        under25_ind.append(i) 
        print(abstracts[i])
        print(cleaned4[i])
        print("\n")
    elif len(elem) > 200:
        over200 += 1
        
        

print("There are %d documents with length < 25" % len(under25))
print("There are %d documents with length > 200" % over200)
        
print("-------------------------------------")
print("Done!")