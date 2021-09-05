# -*- coding: utf-8 -*-
"""

This file contains various functions meant to prepare abstracts for topic
modelling 

Author: Kyla Chasalow
Last edited: September 2, 2021

*get_wordnet_pos() and  word_lemmatizer() taken from code shared with me by
my supervisor/collaborator, Charles Rahal, and written by him 
for "A scientometric review of genome-wideassociation studies"

"""

#Imports
import re
import os
import numpy as np
import numpy_indexed as npi
import copy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle

#language processing
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

#for bigrams
from gensim.models import Phrases  #https://radimrehurek.com/gensim/models/phrases.html
from gensim.corpora.dictionary import Dictionary

#from other scripts
from Helpers import filename_resolver
from Helpers import figure_saver
import Groupbox


#### PRE-PROCESSING ROUND 0 
# remove word or phrases from every abstract - particularly copyright statements
# e.g. "Population Association of America"  from every Demography abstract


def remove_phrase(abstract_list, remove_list = []):
    """
    Remove words or phrases contained in remove_list from every abstract
    note: must match exactly- spacing and capitalization sensitive because
    might want to remove certain proper nouns without removing lower case
    versions 

    if given remove_list = [], does nothing and returns abstract_list    

    Parameters
    ----------
    abstract_list : list of strings, can be pandas series as well but 
        in that case must not have any nan objects
    remove_list : list of strings, optional
        The default is [].

    Returns
    -------
    abstract_list : with all exact match instances of strings in remove_list removed
    """
    if remove_list == []:
        return abstract_list
    else:
        for phrase in remove_list: 
            abstract_list = [abstract.replace(phrase, "") for abstract in abstract_list]
        return(abstract_list)

 
def is_copyright(string):
    """This is only meant to evaluate phrases that come after a © symbol and thus
    have some chance of being a true copyright statement for the abstract.
    Idea is that when a phrase after copyright symbol does not contain these
    words, it is less likely to be an actual copyright statement
    
    If any one of the words below is in the string, returns true
    
    #NOTE: May update this copyright statement identifier list 
    as move to larger data!!!
    
    """
    #checks these sequentially 
    copywrite_words = ["copyright",
                       "rights",
                       "reserved",
                       "university",
                       "association",
                       "organization",
                       "company",
                       "office",
                       "journal",
                       "review" #from annual review
                       ]
    for elem in copywrite_words:
        if elem in string.lower():
            return(True)
    
    return(False)

    
   
def copyright_remover(abstract_list, apply_manual_filter = True):
    """
    Tries to remove as many copyright statements as possible
    
    You are advised to review the output removal list to make sure this has
    resulted only or mainly in actual copyright statements. This will show you whether 
    extra content has been removed BUT it is of course also possible that
    some copyright info has not been removed if it, for example, does not involve
    a © character or involves one but does not end in a period.
    
    FOUR FILTER LEVELS CURRENTLY APPLIED
        1. detect and remove any phrases of the form "Copyright______." where ___ includes no "."
        2. detect and remove any remaining phrases of the form "©_____." where ___ includes no © or .
        3. detect and remove any remaining phrases of the form "All rights reserved"______. where ___ includes no "."
        4. manually remove any remaining instances of "copyright", "© copyright", 'rights reserved',
            and their capitalized variations   
    
    Note that order matters here. (1) is first because if remove a phrase like
        'Copyright © 1999 by Annual Reviews.' first, then the © gets removed, too, but
        if remove the '© 1999 by Annual Reviews' first, then you are left with "Copyright"
        and potentially, issues with there no longer being a phrase of form "Copyright___." 
    
    Parameters
    ----------
    abstract_list : list of strings
    
    apply_manual_filter : bool
        if True, applies is_copyright() function above 
        this is a more conservative option (leads to potentially fewer removals)
        because after trying to detect copyright phrases as described above,
        it only removes phrases that meet a set of manual criteria for being
        copywrite-statement-like

    Returns
    -------
    0. list of abstracts (strings) with copyright removed
    
    1. list of phrases removed

    2. list of any phrases not removed because of manual filtering 
        (if apply_manual_filter = True, else this is an empty list)

    """
    #find all instances of "Copyright____.
    remove_list0  = [c for abstract in abstract_list for c in re.findall(r'[Cc]opyright[^.]*\.', abstract)]
    
    #find all instances of ©_____.  where _____ does not include another © or a "."
    #may miss some copyright info if there are periods in copyright statement
    #but avoids going on for sentences and sentences
    remove_list1  = [c for abstract in abstract_list for c in re.findall(r'©[^©.]*\.', abstract)]
    
    #Also find all instances of "All rights reserved________. where ______
    #does not include another "."
    remove_list2  = [c for abstract in abstract_list for c in re.findall(r'[Aa]ll [rR]ights [rR]eserved[^.]*\.', abstract)]    

    #join them
    remove_list = remove_list0 + remove_list1 + remove_list2
    
    #note: order could matter here - want part below extra so that it first looks to remove phrases
    #above - this is just some more ad-hoc extra
    
    #some manual phrases to remove
    #manually add "Copyright" and "copyright" to list since occasionally
    #have things like © Copyright ©2014 by Annual Reviews. which prevents 
    #"Copyright" from being removed
    remove_list.append("© Copyright")
    remove_list.append("Copyright")
    remove_list.append("© copyright")
    remove_list.append("copyright")
    remove_list.append("rights reserved")
    remove_list.append("Rights Reserved")
    remove_list.append("Rights reserved")
    
       
    if not apply_manual_filter:
        removed = remove_phrase(abstract_list, remove_list) 
        not_removed = []
        return(removed, remove_list, not_removed)
    else:
        #additional manual filtering
        remove_list_filtered = [elem for elem in remove_list if is_copyright(elem)]    
        #get difference
        not_removed = set(remove_list).difference(remove_list_filtered)
        #apply removal list to abstracts
        removed = remove_phrase(abstract_list, remove_list_filtered) 
        return(removed, remove_list_filtered, not_removed)


    
#functions for exploring results of copyright filtering

def count_copyright_symbols(abstract_list):
    """ Counts number of abstracts that contain a
    copyright symbol and  returns their indices. Mostly to be used as a check
    that after "remove_copyright", there are no more
    copyright symbols (or if there are, to help investigate why)
    
    Parameters
    ----------
    abstract_list : list of lists of strings
        
    Returns
    -------
    0. the # of copyright symbols
    1. the indices of abstracts containing copyright symbols
    2. the abstracts containing copyright symbols
    
    
    """
    assert type(abstract_list) == list, "abstract_list must be a list"
    count = 0
    copyright_ind = []
    for i, a in enumerate(abstract_list):
        if str(a) != "nan":
            if "©" in a:
                count += 1
                copyright_ind.append(i)
                
    #get abstracts containing copyright symbol
    copyright_abs = [None] * len(copyright_ind)
    for i, ind in enumerate(copyright_ind):
        copyright_abs[i] = abstract_list[ind]
                
    return(count, copyright_ind, copyright_abs)    
    


def summarize_copyright_list(remove_list):
    """remove_list is list of removed copywrite phrases as 
    returned by copyright_remover. Function removes non-alphabetical
    symbols and then returns list of unique phrases left over. To be used
    for assessing how well copyright removal has worked (e.g. are there
    any phrases of content removed?)
    
    Note that phrases returned are not the exact phrases removed
    This function is meant for exploration/assessment after removal
    to get a sense of what kinds of copyright statements removed
    """
    org_list = [re.sub(r'[\W_]+', ' ', text) for text in remove_list]
    org_list = [re.sub(r'\d+', '', text) for text in remove_list]
    return(np.unique(org_list))


    

#### PRE-PROCESSING ROUND 1

#Helper functions for lemmatization
def get_wordnet_pos(word):
    '''tags parts of speech to tokens
    Expects a string and outputs the string and
    its part of speech'''

    tag = nltk.pos_tag([word])[0][1][0].upper() #this selects just the first letter of POS code
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN) #if tag is not in tag_dict, returns noun as default

def word_lemmatizer(text):
    '''lemamtizes the tokens based on their part of speech'''
    lemmatizer = WordNetLemmatizer()
    text = lemmatizer.lemmatize(text, get_wordnet_pos(text))
    return text


def create_stop_list(custom_stop_list=[]):
    """

    Parameters
    ----------
    custom_stop_list : lst
        list containing any words to be added to stop list. 
        default is to return only the stopwords list contained in the
        nltk english stopwords corpus

    Returns
    -------
    """
    custom_stop_list = [elem.lower() for elem in custom_stop_list] #make lowercase
    stop_list = nltk.corpus.stopwords.words('english')
    #set operation used to ensure get no repeats 
    final_stop_list = list(set(stop_list + custom_stop_list)) 
    return(final_stop_list)


def basic_clean_one_doc(text, stoplist):
    """
    Parameters
    ----------
    text : str
        a string of text
        
   stoplist: lst
        list containing words to remove. use create_stop_list function to access
        the stopwords list contained in the nltk english stopwords corpus and
        possibly to add custom words


    Returns
    -------
    list of tokens which reflects the following preprocessing steps
    
        0. create a list of stopwords, possibly adding custom words to default
        1. remove all but alphanumeric characters
        2. remove all numbers
        3. make all lowercase
        4. obtain lists of words (using whitespace as separator)
        5. lemmatize (obtain 'lemmas' - core root would find in dictionary,
                      e.g. grazing -> graze; women -> woman)
        6. Remove stopwords
        7. Remove any lingering 1 or 2 letter words

    """
    text = re.sub(r'[\W_]+', ' ', text) #remove all but alphanumeric characters, including extra spacing but keeping a space between words
    text = re.sub(r'\d+', '', text)   #remove numbers
    text = text.lower() #make all lowercase
    tokens = word_tokenize(text) #obtain list of words 
    tokens = [word_lemmatizer(w) for w in tokens] #lemmatize
    tokens = [t for t in tokens if t not in stoplist] #remove stopwords
    tokens = [word for word in tokens if len(word) >= 3] #remove <3 letter words
    
    return(tokens)




def basic_cleaner(abstract_list, stoplist, save = False,
                  filename = "cleaned_basic", outpath = None,
                  override = False):
    """
    Applies basic_clean_one_doc to every abstract in abstract_list

    Parameters
    ----------
    abstract_list : list or pandas series
    
    stoplist : list of stop words to remove

    saving options: if save = True, will save file with name filename.pickle
    either in working directory or directory specified by outpath. If override is False,
    it will avoid writing over any existing files with the same name and add _1, _2 etc.
    until it finds a name that does not already exist in target directory

    Returns
    -------
    nested list of cleaned abstracts
    
    
    """
    out_list = [basic_clean_one_doc(text, stoplist) for text in abstract_list]
    
    if save:
        if not override:
            filename = filename_resolver(filename, 
                                         extension = "pickle",
                                         outpath = outpath)
        if outpath is not None:
            filename = os.path.join(outpath, filename + ".pickle")
        else:
            filename = filename + ".pickle"
            
        with open(filename, "wb") as fp:   #Pickling
            pickle.dump(out_list, fp)
    
    return(out_list)
   
    
   
    
   
    
   

### PRE-PROCESSING ROUND 2: DEAL WITH ABVERBS
###
### process does not seem to catch some adverbs that would make
### sense to join (e.g. wise and wisely)
###
### Intuition is to find all words ending in "ly", check whether the stem
### with "ly" removed is also in the vocabulary, and if so,
### replace all the "ly" words with their stem. This avoids
### replacing a word like "homophily" with "homophi" because
### "homophi" will most likely not be in the vocabulary 
###
### However this does not work perfectly and I am not sure yet whether
### I want to take this pre-processing step. It results in changing
### "early" to "ear" and "relatively" to "relative" and the differences
### there can be quite meaningful! There's a trade-off here. Not
### replacing some of the ly's could make a concept seem less prevalent
### than it is but replacing may create falsely high frequencies for 
### certain words (like relative). 
###
### Update: I have added a list of some exceptions to ignore here
### but it's not exchaustive and still a pretty ad-hoc method 
### used https://www.thefreedictionary.com/words-that-end-in-ly#w4
###
### that said, helper functions are written to be general so could be
### used for other corrections


#Function for exploring "ly" word situation beforehand
def ly_summary(abstract_list):
    """
    Summarize the words ending in "ly" in the vocabulary of abstract_list

    Parameters
    ----------
    abstract_list : list of lists of strings

    Returns
    -------
    dictinary with keys:
        
        ly_list : all the words ending in ly in the vocabuary
        stem_in_vocab : list of all stems of words in ly_list for which stem
            is also in vocab
        stem_not_in_vocab : list of all stems of words in ly_list for which
            stem is not in vocab 
            
    *when running processes below, only words with stem in vocab (and which
      are not on exception list) get stemmed

    """
    vocab = extract_vocab(abstract_list)
    #get words ending in ly
    ly_list = [v for v in vocab if re.search(r'ly$',v)]
    #get stems of words ending in ly
    stem_ly = [stem for word in ly_list for stem in re.findall(r'(.+)ly$', word)]
    #get stems in vocab
    stem_in_vocab = [w for w in stem_ly if w in vocab]
    #get stems not in vocab
    stem_not_in_vocab = [w for w in stem_ly if w not in vocab]
    
    out_dict = {}
    out_dict["ly_list"] = ly_list
    out_dict["stem_in_vocab"] = stem_in_vocab
    out_dict["stem_not_in_vocab"] = stem_not_in_vocab
    return(out_dict)




def abstract_replace(abstract, target, new):
    """
    Parameters
    ----------
    abstract : list
        a list of strings
    target : str
        string within abstract to be replaced
    new : str
        string to replace target with

    Returns
    -------
    abstract with all instances of target replaced with new. 
    if target never occurs in abstract, returns original abstract

    """
    new_abstract = [new if word == target else word for word in abstract]
    return(new_abstract)



def abstract_replace_all(abstract, replace_dict):
    """

    Parameters
    ----------
    abstract : list
        a list of strings
    replace_dict : dict
        dictionary containing entries target:new

    Returns
    -------
    abstract with all instances of keys from replace_dict
    replaced by corresponding value

    """
    for elem in replace_dict:
        abstract = abstract_replace(abstract, elem, replace_dict[elem])
    return(abstract)



def fix_adverbs(corpus, save = False,
                  filename = "cleaned_ly", outpath = None,
                  override = False):
    """
    
    Extracts vocabulary from corpus (which should already have been
    pre-processed to lemmatize, remove stopwords etc.). Finds any
    words in vocabulary ending in in "ly" and checks whether stem 
    with "ly" removed is also in vocabulary. For those "ly" instances
    where this is true, replaces "ly" with the stem for all instances
    occuring in all abstracts
    
    E.g. environmentally is replaced with environmental if
    environmental is also in vocabulary but left as is otherwise
    
    E.g. homophily is not changed to homophil unless homophil
    is in the vocabulary (usually would not be)
    
    Parameters
    ----------
    corpus : nested list
        list of lists of strings (aka list of documents)

    saving options: if save = True, will save file with name filename.pickle
        either in working directory or directory specified by outpath. If override is False,
        it will avoid writing over any existing files with the same name and add _1, _2 etc.
        until it finds a name that does not already exist in target directory

    Returns
    -------
    nested list as described above

    """
    #extract vocabulary   
    vocab = extract_vocab(corpus)
    
    # generate necessary replacement dictionary of adverbs from vocab
    ly_list = [v for v in vocab if re.search(r'ly$',v)] #list of all words ending in "ly"
    cut_ly = [stem for word in ly_list for stem in re.findall(r'(.+)ly$', word)] #stems without ly
    cut_in_vocab = [w for w in cut_ly if w in vocab] #find which of the stems also in vocab
    to_replace = [w + "ly" for w in cut_in_vocab] #generate list of "ly" words to replace
    replace_dict = {to_replace[i]: cut_in_vocab[i] for i in range(len(to_replace))}
    
    # remove a few more egregious exceptions (cases where "ly" form has different meaning
    # and both could  be in vocabulary with different meanings)
    # from replacement dictionary 
    # [NOT EXHAUSTIVE]
    
    exceptions = ["apply", "appropriately", 
                  "artificially","barely","childly","closely","comply","courtly",
                  "chiefly","currently","deadly", "deeply","evenly","exactly",
                  "early","fairly", "firmly", "finally",
                  "generally","grossly","hardly","highly","ideally","imply",
                  "intently","integrally","jointly","lastly","lightly","lively","lowly",
                  "likely", "markedly", 
                  "majorly","mainly","marginally","merely","namely","naturally",
                  "mutually","nearly","orderly","oddly","presently","partly",
                  "principally","publicly", "potentially",
                  "purposely", "quarterly","rarely","really", "relatively",
                  "roughly", "seemly", "shortly","socially","solely",
                  "squarely", "strongly","widely","weakly","unlikely","usually"]
    
    for word in exceptions:
        try:
            del replace_dict[word]
        except KeyError:
            pass
    
    #do replacement for each abstract
    new_abstracts = [abstract_replace_all(abstract, replace_dict) for abstract in corpus]
    
    
    if save:
        if not override:
            filename = filename_resolver(filename, 
                                         extension = "pickle",
                                         outpath = outpath)
        if outpath is not None:
            filename = os.path.join(outpath, filename + ".pickle")
        else:
            filename = filename + ".pickle"
            
        with open(filename, "wb") as fp:   #Pickling
            pickle.dump(new_abstracts, fp)
    
    
    return(new_abstracts)



def test_adverb_functions():
    #test replace
    print("Testing: abstract_replace")
    text = "the fish sang of romance la la la".split()
    output = abstract_replace(text, "romance","destruction")
    assert output == "the fish sang of destruction la la la".split(), "error 1"
    print("Text is \n", text)
    print("Output is \n", output)
    print("abstract_replace() appears to be working correctly")

    print("-----")
    #test replace all
    print("Testing: abstract_replace_all")
    output = abstract_replace_all(text, {"la":"bah","fish":"sheep"})
    assert output == "the sheep sang of romance bah bah bah".split(), "error 2"
    print("Text is \n", text)
    print("Output is \n", output)
    print("abstract_replace_all() appears to be working correctly")

    print("-----")
    #test fix adverbs
    print("Testing: fix_adverbs")
    text1 = "greenly green red".split()
    text2 = "green red".split()
    text3 = "green greenly pinkly".split()
    corpus = [text1, text2, text3]
    output = fix_adverbs(corpus)
    assert set(extract_vocab(output)) == set(["pinkly","red","green"]), "error 3"
    print("Corpus is \n", corpus)
    print("Output is \n", output)
    print("fix_adverbs() appears to be working correctly")






### PREPROCESSING ROUND 3: Replace frequent bigrams
###
### 0. extract bigrams occuring over <thresh> times in the dataset 
###
### Two approaches to incorporating bigrams:
###
### 1. bigram_merger() replace them in each document where they occur
###
### for example, ["african","american"] might become ["african_american"]
### but ["elderly","american"] might be left as is. 
### this inflates the size of the vocabulary unless many of the bigrams are such
### that those two words ONLY ever occur together
### 
### Pro: in some cases, it doesn't really make sense to have the bigram components
### as separate words. "african" and "american" mean different things from "african american"
###
### Con: in some cases, it does make sense to have the bigram components - e.g. 
### "result_indicate" gets identified as a bigram in the demography data but result
### and indicate are both related, too. That said, "result" and "indicate" might not be
### very meaningful on their own and making "result_indicate" its own distinct token might
### also then allow us to isolate uses of "result" or "indicate" in other contexts...
###
### 2 bigram_adder() 
###
### this alternative approach appears in LDA tutorial here
### https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html#sphx-glr-auto-examples-tutorials-run-lda-py
### they add the bigrams to the document but leave the original words in, too
###
### Con: this inflates the vocabulary still further and doesn't
### seem to make a lot of sense in some cases. E.g. "machine" and "learning" are related
### but distinct concepts from "machine learning" so if machine learning is what is meant
### it makes sense to just have it read that.
###
### Pro: this approach might be more robust to spurious bigrams and to bigrams
### that also relate to their components... e.g. for life_expectancy or "ethnic_group", you might want to
### keep the words "life," "ethnic," and "group" separate
###
### both approaches are available below
### note that bigram_adder is slower


# NOTE: 
# see here for more on Phrases model used to identify bigrams
# https://radimrehurek.com/gensim/models/phrases.html
# I use default  (scoring = "default") "PMI-like scoring as described 
# in Mikolov, et. al: “Distributed Representations of Words and Phrases 
# and their Compositionality”. https://arxiv.org/abs/1310.4546 


def bigram_merger(abstracts_list, thresh, save_docs = False,
                  doc_filename = "cleaned_bigram_merger", doc_outpath = None,
                  save_bigrams = False, bigram_filename = "bigrams_list", bigram_outpath = None,
                  override = False):
    """

    Parameters
    ----------
    abstracts_list : list of lists of strings
        list containing the documents in list form, e.g. as output 
        from other cleaning functions
    
    thresh : int
        merge bigrams that occur thresh or more times in the corpus.
        making this too high will lead to little or no change
        making this too low will lead to too many mergers, including
        bigrams that simply happen to co-occur a few times by chance

    saving options: 
        for documents and bigram list, if save_docs or save_bigrams = True, will save 
        corresponding output with names specified by filename arguments
        either in working directory or directory specified by outpaths. If override is False,
        it will avoid writing over any existing files with the same name and add _1, _2 etc.
        until it finds a name that does not already exist in target directory
    
        parameters for saving processed documents start with "doc"
        parameters for saving bigrams start with "bigram"
        override is shared by both saving processes



    Returns
    -------
    0. list of lists with all bigrams occuring over thresh times in the corpus 
        merged. Does not modify original abstract list
    
    1. a list of the bigrams added (each bigram included once)
    
    E.g. for one entry, 
    ['size', 'united', 'state', 'furthermore', 'implementation']
    becomes:
    ['size', 'united_state', 'furthermore', 'implementation']

    """
    abstracts = copy.deepcopy(abstracts_list)
    
    #identify bigrams
    bigrams = Phrases(abstracts, min_count=thresh)
    bigram_list = list(bigrams.export_phrases().keys())
    
    for i, doc in enumerate(abstracts):
        abstracts[i] = bigrams[doc] 
        
    if save_docs:
        if not override:
            doc_filename = filename_resolver(doc_filename, 
                                         extension = "pickle",
                                         outpath = doc_outpath)
        if doc_outpath is not None:
            doc_filename = os.path.join(doc_outpath, doc_filename + ".pickle")
        else:
            doc_filename = doc_filename + ".pickle"
            
        with open(doc_filename, "wb") as fp:   #Pickling
            pickle.dump(abstracts, fp)
            
    if save_bigrams:
        if not override:
            bigram_filename = filename_resolver(bigram_filename, 
                                         extension = "pickle",
                                         outpath = bigram_outpath)
        if bigram_outpath is not None:
            bigram_filename = os.path.join(bigram_outpath, bigram_filename + ".pickle")
        else:
            bigram_filename = bigram_filename + ".pickle"
            
        with open(bigram_filename, "wb") as fp:   #Pickling
            pickle.dump(bigram_list, fp)
    
    return(abstracts, bigram_list)




def bigram_adder(abstracts_list, thresh, save_docs = False,
                  doc_filename = "cleaned_bigram_adder", doc_outpath = None,
                  save_bigrams = False, bigram_filename = "bigrams_list", bigram_outpath = None,
                  override = False):
    """
    
    Parameters
    ----------
   abstracts_list : list of lists of strings
        list containing the documents in list form, e.g. as output 
        from other cleaning functions
    
    thresh : int
        add bigrams that occur thresh or more times in the corpus.
        making this too high will lead to little or no change
        making this too low will lead to too many additions, including
        bigrams that simply happen to co-occur a few times by chance

    saving options: 
        for documents and bigram list, if save_docs or save_bigrams = True, will save 
        corresponding output with names specified by filename arguments
        either in working directory or directory specified by outpaths. If override is False,
        it will avoid writing over any existing files with the same name and add _1, _2 etc.
        until it finds a name that does not already exist in target directory
    
        parameters for saving processed documents start with "doc"
        parameters for saving bigrams start with "bigram"
        override is shared by both saving processes

    Returns
    -------
    0. list of lists with all bigrams occuring over thresh times in the corpus added
       to each document they occur in (added n times if occur in document n times)
     
    1. a list of the bigrams added (each bigram included once)
       
    E.g. for one entry, 
    ['size', 'united', 'state', 'furthermore', 'implementation']
    becomes:
    ['size', 'united,' 'state', 'furthermore', 'implementation', "united_state"]

    """
    abstracts = copy.deepcopy(abstracts_list)
    
    #identify bigrams
    bigrams = Phrases(abstracts, min_count=thresh)
    bigram_list = list(bigrams.export_phrases().keys())
    
    for i in range(len(abstracts)):
        for token in bigrams[abstracts[i]]: #bigram[doc] does the merger 
            if '_' in token:
                # Token is a bigram, add to document.
                abstracts[i].append(token)         


    if save_docs:
        if not override:
            doc_filename = filename_resolver(doc_filename, 
                                         extension = "pickle",
                                         outpath = doc_outpath)
        if doc_outpath is not None:
            doc_filename = os.path.join(doc_outpath, doc_filename + ".pickle")
        else:
            doc_filename = doc_filename + ".pickle"
            
        with open(doc_filename, "wb") as fp:   #Pickling
            pickle.dump(abstracts, fp)
            
    if save_bigrams:
        if not override:
            bigram_filename = filename_resolver(bigram_filename, 
                                         extension = "pickle",
                                         outpath = bigram_outpath)
        if bigram_outpath is not None:
            bigram_filename = os.path.join(bigram_outpath, bigram_filename + ".pickle")
        else:
            bigram_filename = bigram_filename + ".pickle"
            
        with open(bigram_filename, "wb") as fp:   #Pickling
            pickle.dump(bigram_list, fp)

    return(abstracts, bigram_list)




### PREPROCESSING ROUND 4: Remove low frequency words
###
### OPTION 1: Remove words that are occur x or fewer times in the whole corpus
###
### Option 2: Remove words that occur in x or fewer documents
### 
### I use option 2. Reasoning: if a certain word occurs 20 times within one document
### it's still not informative about topics in general, which are based on co-occurence
### across documents. Option 1 would not remove it but Option 2 would.
###
### Note: because this is the final pre-processing step, these functions
### also return dictionaries and corpus objects needed for LDA input

def filter_by_word_corpus_freq(docs, thresh = 5):
    """
    
    Remove words that occur thresh or fewer times in the whole list of documents
    contained in docs. Then create a gensim dictionary and corpus and return
    updated documents, corpus, and dictionary
    
   Parameters
    ----------
    docs : nested list
        list of lists of strings (aka list of documents)

    thresh : int, optional
        keep all words with thresh or greater frequency in the corpus.
        The default is 5 (which is quite low!).

    Returns
    -------
    0. updated list of documents with words that have frequency < thresh removed
    """
    fdist = corpus_FreqDist(docs)
    keep_list = [w for w in fdist if fdist[w] >= thresh ]
    new_docs = [[word for word in abstract if word in keep_list] for abstract in docs]  
    return(new_docs)
    
    
    







def filter_by_word_doc_freq(docs, no_below = 5, no_above = 1,
                            save_docs = False, doc_filename = "cleaned_docfreq_filter",
                            doc_outpath = None, override = False):
    """
     
    Remove words that occur in fewer than no_below documents in docs
    or optionally also words that occur in more than no_above *PERCENT* of docs
    (default is no upper removal). Returns dictionary and corpus needed for
    LDA input as well as updated list of list of strings representing the
    documents with the removed words removed from each document
    

    Parameters
    ----------
    docs : nested list
        list of lists of strings (aka list of documents)
        
    no_below : int, optional
        remove words that occur in fewer than no_below documents. The default is 5.
        Note this is a count while no_above is a percent
        
    no_above : float or int in [0,1], optional
        remove words that occur in no_above% or more documents. The default is 1.

    Returns
    -------
    updated list of documents with filtering applied
  
    """
    
    # Create a dictionary representation of the documents
    dictionary = Dictionary(docs)
    #original_length = len(dictionary)
    
    #Filtered dictionary
    dictionary.filter_extremes(no_below = no_below, no_above= no_above)
    #new_length = len(dictionary)
    
    #obtain vocabulary in word string form
    vocab = dictionary.token2id.keys()
    
    #remove words that are no longer in the vocabulary from documents
    new_docs = [[ word for word in doc if word in vocab ] for doc in docs]
    
    # Bag-of-words vector representation of the documents.
    #corpus = [dictionary.doc2bow(doc) for doc in docs]
    
    
    if save_docs:
        if not override:
            doc_filename = filename_resolver(doc_filename, 
                                         extension = "pickle",
                                         outpath = doc_outpath)
        if doc_outpath is not None:
            doc_filename = os.path.join(doc_outpath, doc_filename + ".pickle")
        else:
            doc_filename = doc_filename + ".pickle"
            
        with open(doc_filename, "wb") as fp:   #Pickling
            pickle.dump(new_docs, fp)

    return(new_docs)






#--------------------------------------------------------------------------------
### GENERAL FUNCTIONS
#--------------------------------------------------------------------------------

def get_corpus_and_dictionary(docs):
    """
    Given documents in list of lists of strings form (docs), return 
    vector representation of corpus and dictionary for corpus. These
    are required to train gensim LDA models.
    """
    dictionary = Dictionary(docs)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    return(corpus, dictionary)


   

def extract_vocab(lst):
    """
    given output of clean_all_abstracts(), obtains
    vocabulary of all unique words included anywhere in the list
    
    Parameters
    ----------
    abstracts_list : nested list containing lists of strings

    Returns
    -------
    vocabulary - a alphabetically sorted list of unique strings
    """
    flat_list = [item for sublist in lst for item in sublist]
    vocab = sorted(list(set(flat_list)))
    return(vocab)



def corpus_length(corpus):
    """

    Parameters
    ----------
    corpus : nested list
        EITHER: list of lists of strings (aka list of documents)
        OR: list of lists of tuples of ints (aka corpus in vector
                                             representation from gensim)

    Returns
    -------
    number of words in the corpus (including repeats)

    """
    if type(corpus[0][0]) == str: #string represented corpus
        flat_list = [item for doc in corpus for item in doc]
        length = len(flat_list)
        
    elif type(corpus[0][0][0]) == int: #vector represented corpus
        length = np.sum([num[1] for elem in corpus for num in elem])
    
    return(length) 



def corpus_FreqDist(docs):
    """

    Parameters
    ----------
    docs : nested list
        list of lists of strings (aka list of documents)


    Returns
    -------
    overall frequency distribution for the corpus

    """
    flat_list = [item for doc in docs for item in doc]
    return(nltk.FreqDist(flat_list))





def get_abstracts_by_group(abstract_list, group_labels):
    """
    Function to get abstracts grouped by some label (e.g. journal)
    preserves order in which abstracts from each group appear in abstract_list
    

    Parameters
    ----------
    abstract_list : list of lists of strings
    
    group_labels : list of strings of same length as abstract_list
        with corresponding labels for each abstract

    Returns
    -------
    dictionary with each unique label in group_labels as a key and abstracts
    corresponding to it as a value

    """
    assert len(abstract_list) == len(group_labels), "abstract_list and group_labels must have same length"
    labels, grouped_abstracts = npi.group_by(keys = group_labels,
                                             values = abstract_list)
    out_dict = {}
    for i,lab in enumerate(labels):
        out_dict[lab] = grouped_abstracts[i]
        
    return(out_dict)





def get_unique_words(group_dict):
    """
    helper function 
    
    returns a dictionary with same keys as group_dict but containing only the values
    that are unique to that key
    
    
    ----------
    group_dict : dictionary of lists or arrays

    Returns
    -------
    dictionary as described above
    
    
    Example (this function is actually more general than words)
    -------
    >>> test_dict = {}
    >>> test_dict["a"] = [1,2,3]
    >>> test_dict["b"] = [1,2,4, 10]
    >>> test_dict["c"] = [2,3]
    
    >>> get_unique_words(test_dict)

    {'a': set(), 'b': {4, 10}, 'c': set()}
    
    """
    groups = list(group_dict.keys())
    
    out_dict = {}
    for i,main_group in enumerate(groups):
        out_dict[main_group] = group_dict[main_group]
        temp_groups = groups.copy()
        temp_groups.remove(main_group) 
        for other_group in temp_groups:
            #get whatever is in main group's vocab that is not in the other's vocab
            #assymetric, so this doesn't add things from other group that aren't in main group
            out_dict[main_group] = set(out_dict[main_group]).difference(set(group_dict[other_group]))
            
    return(out_dict)  
    

def get_vocab_by_group(abstract_list, group_labels):
    """
    returns vocabulary for each group given in group labels
    
    Parameters
    ----------
    abstract_list : list of lists of strings
    group_labels : list of strings of same length as abstract_list

    Returns
    -------
    dictionary with unique labels in group_list as keys and the
    vocabulary of each group as values

    """
    assert len(abstract_list) == len(group_labels), "length of abstract_list must == length of group_labels"
    group_dict = get_abstracts_by_group(abstract_list, group_labels)
    groups = list(group_dict.keys())
    out_dict = {g:extract_vocab(group_dict[g]) for g in groups}
    return(out_dict)



def get_corpus_size_by_group(abstract_list, group_labels):
    """
    returns corpus size for each group given in group labels

    Parameters
    ----------
    abstract_list : list of lists of strings
    group_labels : list of strings of same length as abstract_list

    Returns
    -------
    dictionary with unique labels in group_list as keys and the
    corpus size (total word count) of each group as values


    """
    assert len(abstract_list) == len(group_labels), "length of abstract_list must == length of group_labels"
    group_dict = get_abstracts_by_group(abstract_list, group_labels)
    groups = list(group_dict.keys())
    out_dict = {g:corpus_length(group_dict[g]) for g in groups}
    return(out_dict)
  


def get_vocab_summary_by_group(abstract_list, group_labels):
    """
    
    returns a summary dictionary of group vocabularies and their
    inter-relationships as described below
    
    Parameters
    ----------
    abstract_list : list of lists of strings
    group_labels : list of strings of same length as abstract_list

    Returns
    -------
    dictionary with following keys:
        
    lengths : a dictionary with length of each group
    overlap_matrix : a numpy symmetric matrix of dimension g x g, where g is # of groups
        the (i,j)^th element is the percentage of words in the overall vocabulary 
        (aka ignoring group labels) that occurs in both the ith group and the jth group
        vocabularies
    matrix_labels : group labels in order that matrix rows and columns represent
    non_overlaps : dictionary as output by get_unique_words 
        keys are the group names
        values are the words that are unique to only that group and do not occur in any
            of the others

    """
    overall_vocab = extract_vocab(abstract_list)
    overall_size = len(overall_vocab)
    
    #get vocabularies for each group 
    vocab_dict = get_vocab_by_group(abstract_list, group_labels)
    groups = list(vocab_dict.keys())
    n = len(groups)
    
    #get length of each vocabulary
    length_dict = {g:len(vocab_dict[g]) for g in groups}
    
    #holders
    out_mat = np.zeros((n,n))
    #overlap_dict = {}
    
    #get matrix of overlap - that is, what % of words in overall vocab that are in both
    #also store vector of overlapping words for each pair
    indices = list(range(n))     #get all pairs of indices
    pairs = [(a,b) for i,a in enumerate(indices) for b in indices[i+1:]]
    for p in pairs:
        intersection = np.intersect1d(vocab_dict[groups[p[0]]], vocab_dict[groups[p[1]]])
        out_mat[p[0], p[1]] = len(intersection)/overall_size
        out_mat[p[1],p[0]] = len(intersection)/overall_size
        #overlap_dict[groups[p[0]] + "/" + groups[p[1]]] = intersection
        
    #get words unique to each vocabulary
    non_overlap_dict = get_unique_words(vocab_dict)
    
    
    out_dict = {}
    out_dict["lengths"] = length_dict
    out_dict["overlap_matrix"] = out_mat
    out_dict["matrix_labels"] = groups
    out_dict["non_overlaps"] = non_overlap_dict
    #out_dict["overlaps"] = overlap_dict
    return(out_dict)
    









#------------------------------------------------------------------------
########## FUNCTIONS TO EXPLORE ABSTRACT LENGTH 
#------------------------------------------------------------------------


### For a single list of abstracts

def abstract_wordcounts(abstract_lst, years = None, plot = False, color = "purple",
                              alpha = .45, nbins = 30, title_label = "",
                              save_fig = False, fig_outpath = None, 
                              fig_name = "abstract_length_plot", dpi = 150,
                              fig_override = False):
    """
    
    if provided with just a list of abstracts, returns abstract lengths in words and optionally,
    plots a histogram of them
    
    if also provided with a list of corresponding years for each abstract,
    plots a grid containing both the overall histogram and 
    a boxplot-by-year graph of the lengths by year
    

    Parameters
    ----------
    abstract_lst : a list of lists of strings, as output by various cleaner functions
    
    years : a list or np array of corresponding years for each abstract
    
    plot : bool, optional, default is False
        if True, plots a histogram of the 
        
    color : str, optional, default is orange
        optionally set color of the histogram
        
    alpha : float, optional, default is .5
        optionally adjust histogram transparency
        
    nbins : int, optional, default is 30
        optionally adjust number of bins in histogram
        
    title_label : str, optional, goes after "Abstract Lengths by Year" 
        e.g. could let title_label = ": Demography" and get "Abstract Lengths by Year: Demography"

    save_fig...fig_override are the standard figure saving options. See 
        fig_saver() function in Helpers.py

    Returns
    -------
    an np.array containing length of each abstract

    """
    lengths = np.array([len(elem) for elem in abstract_lst])
 
    if plot and years is None:
        plt.rcParams.update({'font.family':'serif'})
        plt.figure(figsize = (15,3))
        plt.hist(lengths, bins = nbins, alpha = alpha, color = color)
        plt.title("Abstract Lengths by Year", pad = 15, fontsize = 22)
        plt.xlabel("Word count after processing", fontsize = 18)
        plt.ylabel("Frequency", fontsize = 18)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)

    elif plot and years is not None:
        
        #npi function orders labels from least to greatest automatically
        labels, grouped_lengths = npi.group_by(keys = years, values = lengths)

        plt.rcParams.update({'font.family':'serif'})
        fig = plt.figure(figsize = (16,11))
        gs = gridspec.GridSpec(30,30)
        
        #obs counts over time bargraph
        ax0 = fig.add_subplot(gs[0:10,0:24])
        counts = [len(g) for g in grouped_lengths]

        plt.bar(np.arange(labels[0], labels[len(labels)-1]+1), 
                counts, 
                color = color,
                alpha = alpha,
                width = 0.6)
        plt.title("Number of Observations per Year \n (missing abstracts removed)", 
                  fontsize = 15, pad = 20)
        plt.ylabel("Observation count", fontsize = 18)
        plt.yticks(fontsize = 15)
        plt.xticks(fontsize = 14)
        plt.xlim(left = np.min(labels)-1, right = np.max(labels)+1) #needed so aligns with below
        
        
        #abstract lengths over time boxplot    
        ax1 = fig.add_subplot(gs[14:30,0:24])
        plt.boxplot(grouped_lengths, vert = True, labels = labels,
                   patch_artist = True, boxprops = dict(facecolor = color, alpha = alpha),
                   medianprops = dict(color = "black"),
                   widths = 0.5
                   )
        plt.ylabel("Word count", fontsize = 18)
        plt.xlabel("Year", fontsize = 20)
        plt.yticks(fontsize = 14)
        plt.xticks(fontsize = 11)
        plt.xticks(rotation = 90)
        plt.ylim(-5,)
        plt.title("Abstract Lengths by Year after Processing" + title_label, pad = 20, fontsize = 22)
        
        #histogram of abstract lengths
        ax2 = fig.add_subplot(gs[14:30,24:30], sharey=ax1)
        plt.hist(lengths, bins = 30, orientation = "horizontal", color = color, alpha = alpha)
        plt.xticks(fontsize = 12)
        plt.title("Marginal Histogram", pad = 10, fontsize = 15)
        plt.setp(ax2.get_yticklabels(), visible = False)
        plt.xlabel("Frequency", fontsize = 18)
         
    
        if save_fig:
           figure_saver(fig_name = fig_name, 
                     outpath = fig_outpath,
                     dpi = dpi,
                     fig_override = fig_override,
                    )   
        
    return(lengths)
                    



# breaking it down by journal -- character counts of abstracts as strings
def abstract_length_by_journal(journals, years, lengths, length_type, color_list = None,
                                  alpha = 0.5, fill_boxes = True, fill_fliers = True,
                                  vert = True, space_between = 1, title = None,
                                  figsize = (20,7), xticks_rotation = 0,
                                  yticks_rotation = 0, xlim = None, ylim = None,
                                  save_fig = False, fig_outpath = None, 
                                  fig_name = "abstract_length_grid", dpi = 200,
                                  fig_override = False):
    """
    

    Parameters
    ----------
    journals : list or array of journal labels
  
    years : list or array of years of same length as journals
    
    lengths : list or array of abstract lengths - word or character counts
    
    length_type : str, either "character" or "word"
    
    color_list : list of strings denoting colors, of same length as number of unique
        journals

    alpha...ylim are plotting parameters. See grouped_boxplots() function in Groupbox.py
        for more information
        
    save_fig...fig_override are the standard figure saving options. See 
        fig_saver() function in Helpers.py


    Returns
    -------
    None.

    """
    options = ["character","word"]
    assert length_type in options, "length_type must be one of " + str(length_type)
    
    grouped_dict = Groupbox.group_values_by_two_dict(outer_variable = years,
                         inner_variable = journals,
                         values = lengths)
    
    #get dictionary in boxplot ready form
    groups, outer_labels, inner_labels = Groupbox.convert_dict_to_lists(grouped_dict)
    
    
    #handle labeling options and things that need to change when switching
    #plot orientation
    if title is None:
        if length_type == "character":
            title = "Abstract Character Counts by Year: Journal Comparison"
        else:
            title = "Abstract Word Counts by Year: Journal Comparison"
    
    min_year = np.min(years)
    max_year = np.max(years)
        
    if vert:
        xlab = "Year"
        if length_type == "character":
            ylab = "Character count"
        else:
            ylab = "Word count"
        if xlim is None:
            xlim = (-1, (max_year-min_year+1)*(len(inner_labels)+space_between)+1)
    
    else:
        ylab = "Year"
        if length_type == "character":
            xlab = "Character count"
        else:
            xlab = "Word count"
        if ylim is None:
            ylim = (-1, (max_year-min_year+1)*(len(inner_labels)+space_between)+1)
    
    
    #plot
    plt.rcParams.update({'font.family':'serif'})
    Groupbox.grouped_boxplots(groups = groups,
                 group_labels = outer_labels,
                 color_labels = inner_labels,
                 color_list = color_list,
                 alpha = alpha,
                 fill_boxes = fill_boxes,
                 fill_fliers = fill_fliers,
                 vert = vert,
                 space_between = space_between,
                 title = title,
                 xlabel = xlab,
                 ylabel = ylab,
                 set_figsize = True,
                 figsize = figsize, 
                 xlim = xlim, 
                 xticks_rotation = xticks_rotation,
                 yticks_rotation = yticks_rotation)

    
    if save_fig:
        figure_saver(fig_name = fig_name, 
                 outpath = fig_outpath,
                 dpi = dpi,
                 fig_override = fig_override,
                 bbox_inches = "tight")

    
