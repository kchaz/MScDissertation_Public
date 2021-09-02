# -*- coding: utf-8 -*-
"""

This file contains functions for processing the output of gensim LdaModels

Author: Kyla Chasalow
Last edited: August 25, 2021


"""
# basics
import os
import numpy as np
import numpy_indexed as npi
import pickle
from scipy.stats import entropy

# language modelling 
from gensim.models.coherencemodel import CoherenceModel
import nltk

# script imports
from Helpers import filename_resolver


# ---------------------------------------------------------------------
# BASICS
# ---------------------------------------------------------------------
def get_top_words(model, topic_id, dictionary, topn = 20):
    """Simple function to get the top <topn> words in string form 
    for topic <topic_id>  from model <model> using dictionary """
    word_tuples = model.get_topic_terms(topicid = topic_id, topn = topn)
    words = [dictionary[elem[0]] for elem in word_tuples]
    return(words)


#--------------------------------------------------------------    
#### Functions to obtain corpus distributions overall and by group
#--------------------------------------------------------------    

def corpus_freq(corpus, dictionary):
    """
    Create a frequency distribution over wordID's in dictionary
    for the entire corpus 

    Parameters
    ----------
    corpus: gensim corpus object
        corpus used to train LDA model (vector-count representation)
        
    dictionary : gensim dictinary 
        dictionary used to train LDA model

    Returns
    -------
    dictionary with word IDs from dictionary as keys and
    counts of each word across entire corpus as values
    """
    #create holder dictionary
    fdist = {i:0 for i in range(len(dictionary))}
    #flatten corpus into one list of (id, counts)
    flat_list = [elem for doc in corpus for elem in doc] 
    #extract info from these tuples
    keys = [elem[0] for elem in flat_list] 
    vals = [elem[1] for elem in flat_list]
    #aggregate counts in dictionary
    for i in range(len(keys)): 
        fdist[keys[i]] += vals[i]
    return(fdist)


def corpus_pdist(freq_dict):
    """
    Normalize corpus frequencies to get empirical distribution
    Also convert from dictionary form to vector form with values in order of
    freq_dict keys, which, if output by corpus_freq, will be in 0...V order
    of vocabulary as encoded by gensim dictionary object
    
    Parameters
    ----------
    freq_dict : dict with counts for each word as output by corpus_freq()

    Returns
    -------
    np.array containing probability distribution (normalized counts)
    for corpus, in same order as dictionary word IDs 0....V
    """
    N = np.sum(list(freq_dict.values()))
    pdist = np.array(list(freq_dict.values()))/N
    return(pdist)
    


def test_fdist(texts, corpus, dictionary):
    """Test to make sure I created fdist correctly """
    #ground truth - getting word frequencies from the actual documents
    fdist_truth = nltk.FreqDist([word for doc in texts for word in doc])
    #function that gets word frequencies using the corpus and dictionary
    fdist_new = corpus_freq(corpus, dictionary)
    errors = 0
    for elem in fdist_truth.keys():
        if fdist_truth[elem] != fdist_new[dictionary.token2id[elem]]:
            errors += 1
    assert errors == 0, "Error, not all frequencies match"



# #no equivalent test ^^^ available for pdist but informally,
# # used this kind of approach to 'sanity' check that high frequency words
# # also have high probability...and that it tracks roughly with topics (though shouldn't be perfect)
# topic = model.get_topics()[3,:]
# print("index",dictionary2.token2id["population"])
# print(fdist[145])
# print(topic[145])
# print(pdist[145])
# print("index",dictionary2.token2id["indian"])
# print(fdist[1758])
# print(topic[1758])
# print(pdist[1758])

def get_corpus_bygroup(corpus, group_list):
    """
    Get group-specific corpora for each group in group_list

    Parameters
    ----------
    corpus: gensim corpus 
        corpus used to train LDA model (vector-count representation)
    
    group_list : list of same length as corpus with group label for each document
     
    Returns
    -------
    dictionary where keys are the unique labels in group_list and
    values are the documents (vector-count representation) for each group
    """
    assert len(corpus) == len(group_list), "corpus length and group_list length must match"
    labels, grouped_corpus = npi.group_by(keys = group_list, values = corpus)
    return({group:corp for (group,corp) in zip(labels, grouped_corpus)})
    


def corpus_freq_bygroup(corpus, dictionary, group_list):
    """
    Get corpus frequency distribution by group

    Parameters
    ----------
    corpus: gensim corpus
        corpus used to train LDA model (vector-count representation)
        
    dictionary : gensim dictinary 
        dictionary used to train LDA model

    group_list : list of same length as corpus with group label for each document
        
    Returns
    -------
    dictionary where keys are the unique values of group_list and
    values are dictionary objects with word_IDs in vocabulary as keys and
    group-specific word frequencies as values
    """
    assert len(corpus) == len(group_list), "corpus length and group_list length must match"
    corpus_dict = get_corpus_bygroup(corpus, group_list)
    groups = np.unique(group_list)
    out_dict = {g:corpus_freq(corpus = corpus_dict[g],
                               dictionary = dictionary) for g in groups} 
    return(out_dict)



def corpus_pdist_bygroup(corpus, dictionary, group_list):
    """
    Get corpus empirical probability distribution by group 
    (normalizing within each group)

    Parameters
    ----------
    corpus: gensim corpus object
        corpus used to train LDA model (vector-count representation)
        
    dictionary : gensim dictinary 
        dictionary used to train LDA model

    group_list : list of same length as corpus with group label for each document
        
    Returns
    -------
    dictionary where keys are the unique values of group_list and
    values are numpy arrays representing probability distribution over the overall
    vocabulary for each group. Note that order is maintained - i.e. the vocab 
    word corresponding to the i^th entry of each array is given by dictionary[i]
    """
    freq_dicts = corpus_freq_bygroup(corpus = corpus,
                                  dictionary = dictionary,
                                  group_list = group_list)
    
    groups = list(freq_dicts.keys())
    p_dists = {g:corpus_pdist(freq_dicts[g]) for g in groups}
    return(p_dists)






#--------------------------------------------------------------    
#### PER-TOPIC METRICS
#--------------------------------------------------------------

def percent_high_freq(model, fdist, topn, thresh):
    """
    A per-topic metric: Examines what percentage of the topn words in each topic
    are high-frequency words -- meant to aid answering questions
    such as: 
    - "are topics just a reflection of corpus frequencies?"
    - "are the top words in the topics generally high frequency words?"
    - "how do these things change over epochs?"

    Parameters
    ----------
    model : gensim LdaModel object
        lda model trained on corpus
        
    fdist : dictionary
        overall corpus frequency counts for each word in the corpus
        as output by corpus_freq()
        
    topn : int
        number of words to consider in each topic
        
    thresh : int in (0,1)
        function will calculate % of words that come from
        top thresh% of corpus by comparing to (1-thresh) quantile

    Returns
    -------
    0.  np.array of length model.num_topics with percentage
        of topn words in each topic that come from top 100*thresh%
        most frequent words in corpus. Order is in terms of topic ID

    1. the frequency cut-off that these percentages represent

    Example:
        suppose top 4 words in topic 0 are
        ["model","data","population","markov"] 
        suppose 3 out of 4 are from the top 10% of words
        in the corpus. Then first element of output array
        with topn = 4 and thresh = .10 would be .75
    """
    #extract IDs from each topic, in order of topic index
    topic_word_IDs = [[elem[0] for elem in model.get_topic_terms(topicid = i,
                                                                 topn = topn)] for i in range(model.num_topics)]
    
    #look up frequency of each word in each topic
    freq_list = [[fdist[word] for word in lst] for lst in topic_word_IDs]
    
    #cut point
    vals = np.array(list(fdist.values())) #get all the counts
    cut = np.quantile(vals, 1-thresh) #thresh % of words are more frequent than this

    #identify which words have frequencies greater than cut in the topics
    sums = np.array([[1 if num > cut else 0 for num in lst] for lst in freq_list])
    
    #calculate percent > cut for each topic
    percents = np.sum(sums, axis = 1)/topn
    return(percents, cut)



def _model_topic_summary(model, metric, corpus, dictionary, 
                         fdist, topn_coherence = 10, 
                         topn_phf = 25, thresh = 0.01
                         ):
    """
    ***meant as internal helper function for topic_summarizer**

    Computes topic-specific metrics for all topics in model and outputs an array
    containing these summaries.For example, for a 10-topic model and
    metric = "coherence", would output a length 10 array of coherences
    
    Summary Metrics
    ----------
        coherence:  'u_mass' coherence of each topic computed using topn_coherence words
        
        phf: stands for "percent high frequency." This is calculated via the
             percent_high_freq() function. It is the percentage of topn_phf
             words in each topic that come from the top (1-thresh)% of words
             by frequency in the overall corpus. Used to examine whether
             topics are mainly relying on high-frequency words
   
        kl: Kullback-Leibler divergence between the topic 
            and the overall corpus. Larger values mean topic is more distinct
            from corpus-wide probabilities of each word. 0 means they are identical.

        entropy: the entropy (H = -sum_i p_i * ln(p_i)) of each topic. Larger
            means more diffuse.
   
    Parameters
    ----------
    model_list : list of gensim LDA models
         list of models trained using Gensim LDA
         Ok to have different num_topics 
        
    metric : str
        must be one of ["coherence","phf","entropy", "kl","all"]

    corpus: gensim corpus 
        corpus used to train LDA model (vector-count representation)
        
    dictionary : gensim dictinary 
        dictionary used to train LDA model
    
    fdist : dictionary
        overall corpus frequency counts for each word in the corpus as output by
        corpus_freq(). Included here as argument rather than calculating it from corpus
        and dictionary so that don't re-calculate it each time in topic_summarizer()
        
    topn_coherence : int, optional, default 10
        # of words to consider when calculating coherence
        
    topn_phf : int, optional, defualt = 25
        # of words to consider when calculating phf
 
    Returns
    -------
    np.array containing values of metric for each of the K topicss in model
    
    If "all", then array contains all metric values in (4 x K) array.
    Row order:
        0. coherences
        1. entropies
        2. KL divergences
        3. phf

    order is always by topic ID: 0...K
    """
    options = ["coherence","phf","kl","entropy", "all"]
    assert metric in options, "metric must be one of" + str(options)
  
    #coherence
    coherences = np.array(CoherenceModel(model=model, corpus=corpus, topn = 10,
                            dictionary=dictionary, coherence='u_mass').get_coherence_per_topic())
    if metric == "coherence":
        return(coherences)
    
    #entropy
    topics = model.get_topics()
    entropies = entropy(topics.T) #topics down columns when do transpose
    if metric =="entropy":
        return(entropies)
    
    #KL
    pdist = corpus_pdist(fdist)
    KL_divergences = [entropy(pk = phi, qk = pdist) for phi in topics]
    if metric == "kl":
        return(KL_divergences)
    
    #phf
    phf = percent_high_freq(model = model, fdist = fdist, topn = topn_phf, thresh = thresh)[0]
    if metric == "phf":
        return(phf)
    
    out_arr = np.array([coherences, entropies, KL_divergences, phf])
    return(out_arr)


def topic_summarizer(model_list, metric, corpus, dictionary, 
                     topn_coherence = 10, 
                     topn_phf = 25, thresh = 0.01,
                     save_array = False, outpath = None,
                     filename = "topic_summary_array",
                     override = False):
    """
    function to obtain model summaries as output by _model_topic_summary()
    for every model in model_list. See details in _model_topic_summary()
    documentation. Returns list containing topic summary for each model in list
   
    Optionally, can SAVE the output array by setting save_array = True.
        Can specify outpath to save it to (else uses working directory),
        q filename to save it with (default 'topic_summary_array'),
        and whether or not to override existing files of same name
        (override = True or False)
    """
    assert type(model_list) == list, "model_list must be a list containing at least one model"
    
    fdist = corpus_freq(corpus, dictionary)
    out_arr = [_model_topic_summary(model = m, 
                                    metric = metric,
                                    corpus = corpus, 
                                    dictionary = dictionary, 
                                    fdist = fdist, 
                                    topn_coherence = topn_coherence,
                                    topn_phf = topn_phf,
                                    thresh = thresh) for m in model_list]
    if save_array:
        if not override:
            filename = filename_resolver(filename = filename,
                                       extension = "pickle",
                                       outpath = outpath)
        if outpath is not None:
            filepath = os.path.join(outpath, filename + ".pickle")
        else:
            filepath = filename + ".pickle"
            
        with open(filepath, "wb") as handle:   
            pickle.dump(out_arr, handle) 
           
    return(out_arr)
    
  
      
                
    
#--------------------------------------------------------------    
##### Theta Matrix, Doc Lengths, Expected word-topic counts
#--------------------------------------------------------------


def _get_doc_wordtopic_mat(lst, K, V):
    """
    helper function for get_document_matrices() which converts a list of the form
    
    [(word_id, [(topic_id, expected_count),(topic_id, expected_count)...]),
     (word_id, [...]),
     ...]

    as output by gensim get_document_topics function for a SINGLE document
    when per_word_topics = True (in particular, the third list output by that function) 
    into a matrix of dimension K x V where K is the number of topics and V
    the number of words in the vocabulary. Many of the entries of this matrix will be 0,
    but those that are not represent expected counts for word v from topic 
    k in a given document        
                
                
    Parameters
    ----------
    lst : list of tuples as described above, represents one document
    K : number of topics
    V : number of words in vocabulary

    Returns
    -------
    a K x V numpy array representing expected word-topic counts for words in
    a single document

    """
    out_mat = np.zeros((K,V))
    for entry in lst: #for each entry of form (word_id, [ (topic_id, expected count), ...])
        word_id = entry[0]
        for topic_tuple in entry[1]: #get list of topics + expected counts
            out_mat[topic_tuple[0], word_id] = topic_tuple[1] #fill into matrix in (topic_id, word_id) slots
    return(out_mat)




def get_document_matrices(model, corpus, minimum_probability = .01, 
                     dictionary = None, per_word_topics = False, minimum_phi_value = 0.01, 
                     save_theta_matrix = False, theta_filename = "theta_matrix", theta_outpath = None,
                     save_wordtopic_arrays = True, wordtopic_filename = "doc_wordtopic_arrays",
                     wordtopic_outpath = None, override = False):
    """
    This function executes two major tasks but can be set to only do the first
    
    TASK 1:
        Obtain K x D matrix of documents' topic probabilities
        Depending on minimum_probability setting, some entries may be rounded 
        down to 0.
    
        Optionally, save the matrix for future use.
    
    Task 2: (only run if per_word_topics = True)
        Obtain  D x K  x V array of each document's expected word counts for each word
        v and topic k. 
        
        That is, values reflect the calculation N_{dv} * \phi_{dvk}  where N_{dv}
        is the number of times word v occurs in document d and \phi_{dvk} is the fitted
        variational parameter for the probability of word v in document d coming from 
        topic k
        
    The reason these two tasks are together in one function is that gensim's .get_document_topics
    function does both.
    
    
    WARNING: Gensim's procedure for obtaining values used here involves
    re-running local document variational inference updates and this involves randomness.
    Each time you run this function, the output may be slightly different. This is one
    motivation for including the save option in this function so that it will be possible to
    recover output from a run of the function
    
    ##TO DO: Figure out how to set random state

    Parameters
    ----------
    model : LDA Gensim model object
    
        note: doesn't appear to matter whether model trained with per_word_topics set to True?

    corpus : list
        corpus (vector representation) model was trained on
   
    dictioanry : dictionary used to train model
        optional because only needed if per_word_topics is True
        
    override : bool
        applies to all saving processes below. If True, will not use filename_resolver
        and will over-write any existing files of same names when saving
   
    THETA MATRIX PARAMETERS
    -----------------------
    minimum_probability : float, optional
        probabilities below this value will be set to 0 in matrix. The default is .01.
        
    save_theta_matrix, theta_filename, theta_outpath control whether and where theta
    matrix is saved and with what name. If override = False, then if filename already 
    exists at save location, it will be amended with _1, _2 etc. 
    until a name is found that does not already exist
        
    WORDTOPIC ARRAY PARAMETERS
    ----------------------------
    minimum_phi_value : expected counts under this value are set to 0
        name is slightly misleading because it doesn't appear to be a cut-off for \phi_{dvk} but
        this is the name gensim uses
        
    save_wordtopic_arrays, wordtopic_filename, wordtopic_outpath control whether and where theta
    matrix is saved and with what name. If override = False, then if filename already 
    exists at save location, it will be amended with _1, _2 etc. until 
    a name is found that does not already exist

    Returns
    -------
    0. num_topics x num_documents matrix where:
        - Each column represents a document (in same order as corpus)
        - Each row represents a topic
        - Values are the probability that a word in document d comes from topic k
 
    1. if per_word_topics = True, also returns a (D x K x V) array as described above

    """
    if per_word_topics:
        assert dictionary is not None, "if per_word_topics = True, dictionary must be given"
        V = len(dictionary)
    
    # Create generator to get info needed for both matrices
    doc_generator = model.get_document_topics(corpus, 
                                          minimum_probability = minimum_probability,
                                          minimum_phi_value = minimum_phi_value,
                                          per_word_topics = per_word_topics)
    
   
    #this part takes some time - actually generate output for each document
    list_of_output = [doc_generator[i] for i in range(len(corpus))]

  
    # CREATE THETA MATRIX
    
    #document-topic-assignment-lists to conver to matrix
    if per_word_topics:
        list_of_doc_topic_assignments = [elem[0] for elem in list_of_output]
    else: 
        list_of_doc_topic_assignments = list_of_output
    
    # Set up holder matrix of 0's for theta matrix
    D = len(corpus)
    K = model.num_topics
    theta_mat = np.zeros((K,D))
    
    #replace values in holder matrix using (topic ID, probability) pairs
    for d, lst in enumerate(list_of_doc_topic_assignments): #list corresponding to each document
        for elem in lst: #each a tuple
            theta_mat[elem[0],d] = elem[1] 

    #saving options
    if save_theta_matrix:
            #if filename already exists ammend _1, _2, etc. until find one that does not
            if not override:
                theta_filename = filename_resolver(theta_filename, extension = "npy", outpath = theta_outpath)
            #amend outpath if given
            if theta_outpath is not None:
                theta_filename = os.path.join(theta_outpath, theta_filename)
            np.save(theta_filename,theta_mat)
    
    #stop here if only asked for theta matrix
    if not per_word_topics: 
        return(theta_mat)
    
    #-------------------------------

    # CREATE PER-WORD-TOPIC MATRICES of expected counts 
    else:    
        #get word-specific topic assignment lists
        list_of_doc_wordtopic_counts = [elem[2] for elem in list_of_output]
        
        #use helper function to generate matrix for each
        doc_wordtopic_arrays = np.array([_get_doc_wordtopic_mat(lst,
                                                               K = K,
                                                               V = V) for lst in list_of_doc_wordtopic_counts])

        if save_wordtopic_arrays:
            #if filename already exists ammend _1, _2, etc. until find one that does not
            if not override:
                wordtopic_filename = filename_resolver(wordtopic_filename,
                                                   extension = "npz",
                                                   outpath = wordtopic_outpath)
            #amend outpath if given
            if wordtopic_outpath is not None:
                wordtopic_filename = os.path.join(wordtopic_outpath, wordtopic_filename)
            #use compressed format because very large!
            np.savez_compressed(wordtopic_filename,doc_wordtopic_arrays)#takes time
            
        return(theta_mat, doc_wordtopic_arrays)
    

def load_wordtopic_array(filename, path = None):
    """little function to help seemlessly load wordtopic array saved as
    .npz file by get_document_matrices()
    
    Note this can take some time
    
    filename is a string with .npz. 
    Assumes only one array has been saved this way
    and returns it
    """
    if path is not None:
        filename = os.path.join(path, filename)
    load_array = np.load(filename)
    return(load_array[load_array.files[0]])





def get_doc_lengths(corpus):
    """
    Parameters
    ----------
    corpus : corpus as used to train LDA Gensim models (vector representation)
 
    Returns
    -------
    a vector containing the length of each document in corpus (in same order as corpus)
    """
    return(np.array([np.sum([entry[1] for entry in elem]) for elem in corpus]))

        


#-------------------------------------------------------------------
#FOUR WAYS TO CALCULATE TOPIC SIZE 
#-------------------------------------------------------------------

def get_topic_wordcounts(theta_mat, doc_lengths, normalized = False):
    """
    Calculates *EXPECTED* topic size in terms of word counts for each topic in the following way:
        
        Let K be number of topics and D be number of documents
        Let M be a K x D be matrix of per-document topic probabilities
        Let N be a D x 1 vector containing each document's length in words (in same
         order as columns of M)
        Then output of this function is   S = MN
        
        This represents   \sum_{d=1}^{D}  \theta_{dk} N_{d}, the sum of the 
        expected number of words from topic k for each document d
                                                                           
    
    WARNINGS:
        * topic sizes will not sum perfectly to the number of words in the original
          corpus because of the use of probabilities here. 
          
        * topic sizes may have decimals (again because using probabilities)
        
        * IF documents in corpus have wildly different sizes, topic sizes may be
          largely impacted by the largest documents in the corpus. To give an extreme example,
          suppose you had a corpus with 1000 short articles about horses and 
          one long book about steam engines. Then the topic size for a steam engine might,
          as calculated above, be very large because a large number of words in the corpus
          are expected to come from it. In one sense, this is correct but in another sense,
          if we drew a document randomly, we'd be unlikely to draw a document about steam engines
          and in that sense, the topic is rare. 
          
          That said, it seems unlikely this will be a major issue for large quantities of documents of roughly same
          order of magnitude size 
          
          See get_topic_doc_counts() for a size calculation that maps more to the latter concern
        

    Parameters
    ----------
    theta_mat : K x D matrix as output by get_document_matrices()
       
    doc_lengths : length D vector as output by get_doc_lengths()
        DESCRIPTION.
        
    normalised : bool, optional
        if true, will normalize topic sizes by dividing by the sum of all sizes.
        These may be easier to interpret. The default is False.
        

    Returns
    -------
    a numpy array of length K containing size of each topic (maintaining order
    given via rows of theta_matrix - should be just 0...K)

    """
    assert theta_mat.shape[1] == doc_lengths.shape[0], "dimension mismatch between theta matrix and doc lengths"
    sizes = np.matmul(theta_mat, doc_lengths)
    if normalized:
        sizes = sizes / np.sum(sizes)
    return(sizes)
  
    
  
def get_topic_doccounts(theta_mat, normalized = False, thresh = None):
    """
    
    Provides an alternative way to think about topic size to get_topic_wordcounts()
    
    Rather than calculate expected counts using calculations like:
            
            (Number of words in document) * (prob of topic in document)
            
    which may sometimes be largely influenced by particularly large documents, 
    
    This function simply calculates the number of documents in which the probability
    of a given topic is not 0 aka the "count" of the documents where the topic occurs

    
    WARNING:
    -----------
    This is dependent on choice of minimum_probability in get_document_matricesrix()
    function used to get theta_matrix. At low extreme, all topics
    will "occur" in all documents (at least if haven't set thresh parameter)
    and at high extreme, no topics will occur in any documents.                                
    

    Parameters
    ----------
    theta_mat : K x D matrix as output by get_document_matricesrix()
    
    thresh : float between 0 and 1, default is None
    
          optionally, "count" a document as containing a topic only if its probability
          for that topic is greater than or equal to thresh. 
          If thresh = None, counts document as containing topic as long as probability 
          for that topic is greater than 0
        
        * will not raise error if outside 0 or 1 but those values will not be
          very meaningful 
  
    normalized : bool, optional
        if True, normalizes counts. The default is False.

    Returns
    -------
    a numpy array of length K containing count for each topic (maintaining order
    given via theta_matrix)
    """

    if thresh is None:
        counts = np.sum(theta_mat != 0, axis = 1)
    else:
        counts = np.sum(theta_mat >= thresh, axis = 1)
    if normalized:
        counts = counts/np.sum(counts)
    return(counts)
        


def get_topic_means(theta_mat, normalized = False):
    """
    Another way of thinking about topic size - mean theta value for each
    topic over documents
    
    Parameters
    ----------
    theta_mat : K x D matrix as output by get_document_matricesrix()
  
    normalized : bool, optional
        if True, normalizes counts. The default is False.

    Returns
    -------
    a numpy array of length K containing arithmetic mean theta value
    for each topic
    """
    means = np.mean(theta_mat, axis = 1)
    if normalized:
        means = means / np.sum(means)
    return(means)
        
def get_topic_medians(theta_mat, normalized = False):
    """
    Another way of thinking about topic size - median theta value for each
    topic over documents
    
    WARNING: Not very useful. Theta matrix is sparse so median
    will often be 0 or close to it
    
    Parameters
    ----------
    theta_mat : K x D matrix as output by get_document_matrices()
  
    normalized : bool, optional
        if True, normalizes counts. The default is False.

    Returns
    -------
    a numpy array of length K containing median theta value
    for each topic
    """
    medians = np.median(theta_mat, axis = 1)
    if normalized:
        medians = medians / np.sum(medians)
    return(medians)




#------------------------------------------------------------------------------
# CALCULATE EXPECTED COUNTS BY WORD
#------------------------------------------------------------------------------

def get_topic_perwordcounts(model, theta_mat, doc_lengths, normalized = False):
    """    
    Calculates the expected count of each word in the vocabulary for each topic.
    That is, using expected topic wordcounts as calculated by get_topic_wordcounts(),
    calculates, for each topic k and word v:
        
            E(N_kv) = E(N_k) * \phi_kv
            
    Optionally, these can be normalized over v. Normalized values will track with 
        but not be identical to the original topic probability matrix phi. 
        In particular, calculations here take into account the (expected) 
        size of each topic while original topic probability matrix does not. 
        If some topics are very small, for example, then E(N_kv) may be relatively 
        smaller than the corresponding value in phi
    
    Parameters
    ----------
    model : Gensim LDA model object
        
    theta_mat : matrix as output by get_document_matrices()
                
    doc_lengths : vector as output by get_doc_lengths()
        
    normalized : bool, optional
        if True, normalizes each row. 

    Returns
    -------
    Matrix of E(N_kv) values, of shape  num_topics x size of vocabulary (K x V)
    Default is for these to be unnormalized but can be normalized

    """
    #Note: below, the [:,None] is a trick so broadcasting works and it
    #multiplies/divides each row by corresponding element of vector
    
    #get topic probabilities
    phi = model.get_topics()
    #get expected word count for each topic
    word_counts = get_topic_wordcounts(theta_mat, doc_lengths, normalized = False)
    #multiply each row of phi by corresponding value of word_counts for each topic
    out_mat = phi * word_counts[:,None]
    assert out_mat.shape == phi.shape, "error: phi and out_mat don't have same shape. Something has gone wrong. It may be that model and word_counts/doc_lengths are not matched"
    
    if normalized:
        out_mat = out_mat/ np.sum(out_mat, axis =1)[:,None]
    
    return(out_mat)



def get_overall_wordcounts(perword_counts_mat):
    """
    perword_counts_mat is output by get_topic_perworcount()
    
    This function sums over k's (rows) to get expected count for each word overall in the corpus"""
    return(np.sum(perword_counts_mat, axis = 0))



         



                         

# RELEVANCE CALCULATIONS FOR OVERALL MODEL
#---------------------------------------------------------
def get_relevance_matrix(corpus, dictionary, model = None,
                         phi = None, pdist = None,
                         lamb = 0.6):
    """
    Calculates the relevance of each word in each topic using following
    formula from Sievert and Shirley paper on LDAviz
    
        relevance = (lambda)log(phi_kv) + (1-lambda) log(phi_kv/p_v)
    
    where p_v is the overall corpus frequency.
    
    The idea is that if a word has high probability in a topic but is also 
    very common overall, it is less distinguishing of the topic and thus has
    lower "lift" (the second term above) and lower relevance

    Note: when the probability of a word in a topic is phi_kv = 0,
        I manually set relevance to -np.inf. Some warning messages may arise from when
        relevance calculation first encounters log(0) or 0log(0). Python automatically
        sets log(0) to -np.inf but if lambda = 1 or 0, 0log(0) gets set to np.nan. This is
        what I set to np.inf, the point being that relevance is as low as possible 

    Note that p_v is never 0 since to be in the vocabulary, a word must occur
        at least once and thus have non-zero empirical probability

    Parameters
    ----------
 
    corpus : corpus used to train Gensim LDA model (vector representation)
    
    dictionary : dictionary used to train Gensim LDA model
    
    
    *note: one of model or phi must be given
    
        model : LDA gensim model
        
        phi: np.array of dimension num_topics x len(dictionary), optional
                used to specify your own topic distributions. 
                
        if phi is None, model must be given and topics from model are used
        if phi is not None, even if model is given, it overrides model and only phi
            is used
            
    pdist : array of length of vocabulary, optional
        optionally, specify own probability distribution over corpus
        to use for relevance calculation. Else, will calculate it from
        corpus. Note that it is assumed to be in order of vocabulary
        as recorded in dictionary indices (1...V)

    lamb : float between 0 and 1, optional
        controls trade-off between word probability and "lift".
        lamb = 0 means we consider only lift (may select very rare words) 
        lamb = 1 means we consider only topic probabilities
        The default is 0.6.
        
    
    Returns
    -------
    0. matrix containing relevance of each word in each topic (num topics x num_words)
    
    1. matrix giving the indices that would sort each row of the relevance matrix from greatest
    to least. E.g. if order_matrix[1,0] = 54, then the 54th word is the word with highest
    relevance in topic 1
    """
    assert lamb >= 0 and lamb <= 1, "lambda must be in [0,1]"
    assert model is not None or phi is not None, "one of model or phi must be given"
    
    #get overall distribution over corpus
    if pdist is None:
        pdist = corpus_pdist(corpus_freq(corpus = corpus, dictionary = dictionary))
    
    #get topics if not given
    if phi is None:
        phi = model.get_topics()
        
    #calculate lift (divides each row of phi by pdist)
    lift_mat = phi/pdist
    #relevance calculation
    relevance_mat = (lamb * np.log(phi)) + ((1 - lamb) * np.log(lift_mat))
    
    #set any nans to -np.Inf since they arise from situation with 0 log(0)
    relevance_mat[np.isnan(relevance_mat)] = -np.Inf
    
    #get order matrix giving indices that would sort each row, greatest to least (using np.flip)
    order_mat = np.flip(np.argsort(relevance_mat, axis = 1), axis =1) 
    
    assert relevance_mat.shape == order_mat.shape, "something has gone wrong - dimension mismatch order_mat and relevance_mat"
    assert relevance_mat.shape == phi.shape, "something has gone wrong - dimension mismatch phi and relevance_mat"
        
    return(relevance_mat, order_mat)



def get_topword_probs(order_mat, corpus, dictionary, model = None, phi = None, topn = 10):
    """
    Get the corresponding words, topic probabilities, and corpus-wide probabilities
    for the topn words of each topic as given by order_matrix
    
    order matrix could, for example, be output by the get_relevance_matrix()
    function and therefore give the words ordered by relevance
    
    Function intended to aid in creating bar plots representing topics

    Parameters
    ----------
    order_mat : np.array of dimensions K x num_words containing indices that
        encode an order of words (e.g. from greatest to least relevance)
        to be used in determining the topn words
        
    corpus : corpus used to train Gensim LDA model (vector representation)
    
    dictionary : dictionary used to train Gensim LDA model
    
     *note: one of model or phi must be given
    
        model : LDA gensim model
        
        phi: np.array of dimension num_topics x len(dictionary), optional
                used to specify your own topic distributions. 
                
        if phi is None, model must be given and topics from model are used
        if phi is not None, even if model is given, it overrides model and only phi
            is used


    topn : int, optional
        number of top words to examine. The default is 10.

    Returns
    -------
    0. list of lists where each list contains the topn words (in order from order_mat)
        for each topic (in topic id order)
    1. list of lists where each list contains the topic probabilities of each of the words in 0.
    2. list of lists where each list contains the overall corpus empirical probabilities of each
       of the words in 0

    """
    assert model is not None or phi is not None, "one of model or phi must be given"
   
    #get topics if not given
    if phi is None:
        phi = model.get_topics()

    #get overall corpus distribution over words
    pdist = corpus_pdist(corpus_freq(corpus = corpus, dictionary = dictionary))
    
    #get top words in each topic using orders (e.g. orders determined by relevance)
    top_word_list = [[dictionary[i] for i in ind[:topn]] for ind in order_mat] 
    
    #get phi probabilities for each list of top words for each topic:
    #For each topic and for each of topn indices corresponding to the topn most
    #relevant words in that topic...use those indices to get the topic probabilities
    #of those words
    top_word_topic_probs = [ [phi[i,:][j] for j in ind[:topn] ] for (i, ind) in enumerate(order_mat)]
    
    #get overall corpus probabilities
    top_word_overall_probs = [[pdist[i] for i in ind[:topn]] for ind in order_mat]
    return(top_word_list, top_word_topic_probs, top_word_overall_probs)




def get_topword_counts(order_mat, model, dictionary, theta_mat, doc_lengths, topn = 10):
    """
    Get the corresponding words, topic expected word counts, and corpus-wide expected word counts
    for the topn words of each topic in model as given by order_matrix
    
    order matrix could, for example, be output by the get_relevance_matrix()
    function and therefore give the words ordered by relevance
    
    Function intended to aid in creating bar plots representing topics


    Parameters
    ----------
    
    order_mat : np.array of dimensions K x num_words containing indices that
        encode an order of words (e.g. from greatest to least relevance)
        to be used in determining the topn words
   
    model : gensim LDA model
        
    dictionary : dictionary used to train Gensim LDA model

    theta_mat : np.array of dimension  num_topics x num_documents, 
        as output by get_document_matrices()

    doc_lengths : np.array of dimension num_documents, as output by
        get_doc_lengths()

    topn : int, optional
        number of top words to examine. The default is 10.
        

    Returns
    -------
    0. list of lists where each list contains the topn words (in order) for each topic as
       given via order_mat
    1. list of lists where each list contains the expected topic wordcounts of each of the words in 0.
    2. list of lists where each list contains the overall expected word count of each
       of the words in 0

    """
    assert order_mat.shape == model.get_topics().shape, "order_mat must match model topics"
    
    #get topic word counts
    perword_count_mat = get_topic_perwordcounts(model, theta_mat, doc_lengths, normalized = False)
    
    #get overall corpus word counts
    overall_counts = get_overall_wordcounts(perword_count_mat)
    
    #get top words in each topic using orders (e.g. orders determined by relevance)
    top_word_list = [[dictionary[i] for i in ind[:topn]] for ind in order_mat] 
    
    #get expected wordcounts for each list of top words for each topic
    top_word_topic_counts = [ [perword_count_mat[i,:][j] for j in ind[:topn] ] for (i, ind) in enumerate(order_mat)]
    
    #get overall corpus probabilities
    top_word_overall_counts = [[overall_counts[i] for i in ind[:topn]] for ind in order_mat]
    
    return(top_word_list, top_word_topic_counts, top_word_overall_counts)
    

    
    
    

#----------------------------------------------
############### BY-GROUP EXPLORATIONS #######
#---------------------------------------------
# These functions make it possible to do many of the things above only dividing
# the documents into groups. The general framework is that the user provides a
# vector containing a label for each document in the corpus
# the functions then use those labels to group the documents



def get_theta_mat_by_group(theta_mat, group_labels):
    """
    Get a dictionary of theta matrices by group

    Parameters
    ----------
    theta_mat : K x D array as output by get_document_matrices
    
    group_labels : length D array of group labels for each document

    Returns
    -------
    dictionary 
        keys are the unique values within group_labels
        values are K x D_g numpy arrays corresponding to the D_g columns
        of the K x D matrix that are in each group g
    """
    #first group indices
    D = theta_mat.shape[1] 
    labels, theta_ind_groups = npi.group_by(keys = group_labels, values = np.arange(0,D))
    #then access elements of matrix
    theta_mat_dict = {group:theta_mat[:,group_ind] for group, group_ind in zip(labels, theta_ind_groups)}        
    return(theta_mat_dict)





def get_per_group_topic_size(theta_mat, label_list, sizetype, doc_lengths = None, normalized = True):
    """
    Uses one of three approaches to topic size to calculate topic sizes for each group 
    defined by label_list. This can be used to compare, for example, topic sizes within 
    different time slices
    
    note: median is not included as an option because usually ends up just being all 0's
    
    Parameters
    ----------
    theta_mat : theta matrix for the entire corpus as output by get_document_matrices
    
    label_list : list of same length as corpus containing labels to be used to sort documents
        note: if all labels are the same, function will still work and output a one-key dictionary 
        
    sizetype : str, one of ["word_count", "doc_count", "mean"]
        
    doc_lengths : np.array or list of doc lengths, required if sizetype = "word_counts"
            and else not used and allowed to leave it None
    
    normalized : bool, optional
        normalizes the topic sizes within each group
        default is True because if groups have different sizes, normalization
        makes them more comparable
        
    Returns
    -------
    dictionary with labels as keys and topic size vectors (in order of topic IDs 0,1...K-1)
    as values

    """
    #robust to receiving a list instead of a np.array for this
    if type(doc_lengths) == list:
        doc_lengths = np.array(doc_lengths)
        
    options = ["word_count","doc_count","mean"]
    assert sizetype in options, "sizetype must be one of" + str(options)
    if sizetype == "word_count":
        assert doc_lengths is not None, "for expected word counts, doc_lengths argument must be specified"
    
    D = theta_mat.shape[1]
    assert len(label_list) == D, "label_list must be same length as number of documents"
    
    #get grouped document indices
    #for int values, automatically sorts them so for years, will always sort the years - useful
    label_order, groups = npi.group_by(keys = label_list, values = np.arange(0,D))
    
    #store group-specific theta matrices and document lengths
    doc_len_dict = {}
    theta_mat_dict = {}
    for label, group_ind in zip(label_order, groups):
        theta_mat_dict[label] = theta_mat[:,group_ind]
        if sizetype == "word_count":
            doc_len_dict[label] = doc_lengths[group_ind]
    
    #calculate group-specific size measures
    size_dict = {}
    if sizetype == "word_count":
        for label in label_order:
            size_dict[label] = get_topic_wordcounts(theta_mat_dict[label], 
                                                              doc_len_dict[label],
                                                              normalized = normalized)
    elif sizetype == "doc_count":
        for label in label_order:
            size_dict[label] = get_topic_doccounts(theta_mat_dict[label],
                                                            normalized = normalized)
    elif sizetype == "mean":
        for label in label_order:
            size_dict[label] = get_topic_means(theta_mat_dict[label], normalized = normalized)
            
    # elif sizetype == "median": #not recommended - doesn't work well, but included so can explore it if want
    #     for label in label_order:
    #         size_dict[label] = get_topic_medians(theta_mat_dict[label], normalized = normalized)
    
    return(size_dict)
    





def get_per_group_topics(doc_wordtopic_arrays, label_list, normalized = False,
                         save_dict = False, filename = "per_group_topic_dict",
                         outpath = None, override = False):
    """    
    doc_wordtopic_arrays contains for each document, the expected number of words 
    observed for each word and topic combination. To get the expected counts
    for a group of documents, these matrices can be summed elementwise. To get the equivalent
    of an actual topic distribution, these then need to be normalized over every row. 
    This function does this, with normalization optional, providing in that way
    'group-specific-topics'

    Parameters
    ----------
    doc_wordtopic_arrays : array of dimension D x K x V as output by 
        get_document_matrices with per_word_topics = True
        
    label_list : list or array of labels of same length as number of documents
    
    normalized : bool, optional
        if True, normalizes each row of each output matrix. The default is False.
    
    save_dict...override : optionally, save output dictionary with given filename,
    using outpath as location if given, and overriding any existing files with same name
    if override is true or otherwise, resolving filename

    Returns
    -------
    dictionary with labels as keys and corresponding group-specific topic 
    arrays as values
    
    *NOTE: it is possible for a group's topic array to contain rows with all 0. If row
    k is all 0's, this indicates that for every document in the group, the expected number 
    of times each word in each document comes from topic k is 0 -- that is, the topic is just
    not present in documents from that group. 
    
    when this 0-row happens, it is possible to get a runrime warning when normalizing:
    "RuntimeWarning: invalid value encountered in true_divide" but note that the
    function deals with resulting nan values from dividing by 0 by setting them to 0.

    """
    assert doc_wordtopic_arrays.shape[0] == len(label_list), "label_list must have same length as number of documents"
    
    #group the arrays using labels
    label_order, groups = npi.group_by(keys = label_list, values = doc_wordtopic_arrays)
    #create a dictionary with labels as keys and corresponding sum over matrices for all groups as value
    #this is an estimate of the per-group topic
    out_dict = {}
    for label, group in zip(label_order, groups):
        mat = np.sum(group, axis = 0)  
        if normalized: #normalize each row
            mat = mat / np.sum(mat, axis = 1)[:,None]
            #if normalization results in any nan, this is because row contains all 0's 
            #and thus there has been division by 0. Line below sets these to 0.
            #row with all 0's can happen if for every document in group, probability of words
            #coming from that topic is 0
            mat[np.isnan(mat)] = 0 
        out_dict[label] = mat
                   
    if save_dict:
        if not override:
            filename = filename_resolver(filename = filename,
                                       extension = "pickle",
                                       outpath = outpath)
        if outpath is not None:
            filepath = os.path.join(outpath, filename + ".pickle")
        else:
            filepath = filename + ".pickle"
            
        with open(filepath, "wb") as handle:   
            pickle.dump(out_dict, handle) 

    return(out_dict)





# RELEVANCE CALCULATIONS FOR LOOKING AT TOPICS BY GROUP - EVERYTHING
# BROKEN DOWN BY GROUP
#---------------------------------------------------------

def get_group_relevance_dict(corpus, dictionary, group_list, per_group_topic_dict,
                             lamb = 0.6):
    """
    Break corpus into groups and get a group-specific relevance matrix
    and corresponding order matrix as output by get_relevance_matrix()    

    Parameters
    ----------
    corpus: gensim corpus 
        corpus used to train LDA model that generated wordtopic_array
        (vector-count representation)
        
    dictionary : gensim dictinary 
        dictionary used to train LDA model that generated wordtopic_array
   
    group_list : length D list of group labels for each document

    per_group_topic_dict : dictionary as output by get_per_group_topics()
    
    lamb : int or float in [0,1], optional
        see get_relevance_matrix(). The default is 0.6.

    Returns
    -------
    dictionary
        keys are the unique values of group_list
        
        values are corresponding output of get_relevance_matrix()
        when applied only to the corpus that comes from each group
        and using that corpus's overall empirical distribution for the
        lift calculation
    
    """
    groups = np.unique(group_list)
    corpus_dict = get_corpus_bygroup(corpus = corpus,
                                   group_list = group_list)
    #get pdist by corpus
    pdist_dict = corpus_pdist_bygroup(corpus = corpus, 
                                      dictionary = dictionary,
                                      group_list = group_list)
    
    #apply relevance function after breaking everything into groups
    out_dict = {}
    for g in groups:
        out_dict[g] = get_relevance_matrix(corpus = corpus_dict[g],
                                           dictionary = dictionary,
                                           model = None,
                                           phi = per_group_topic_dict[g],
                                           pdist = pdist_dict[g],
                                           lamb = lamb) 
    return(out_dict)



def get_topword_values_dict(corpus, dictionary, 
                           group_list, 
                           per_group_relevance_dict,
                           per_group_topic_dict,
                           normalized,
                           topn = 20):
    """
    For each group-specific corpus and each group-specific topic in per_group_topic_dict,
    get the top words, the probabilities/expected counts and the corpus-wide 
    probabilities/corpus-counts 

    Parameters
    ----------
    corpus: gensim corpus 
        corpus used to train LDA model that generated wordtopic_array
        (vector-count representation)
        
    dictionary : gensim dictinary 
        dictionary used to train LDA model that generated wordtopic_array
   
    group_list : length D list of group labels for each document

    per_group_relevance_dict : dictionary output by get_group_relevance_dict

    per_group_topic_dict : dictionary as output by get_per_group_topics()
   
    normalized : bool,
        if False, retrieves expected counts
        if True, retrieves normalized probabilities calculated using expected counts
        
    topn : int, optional
        number of top words to retrieve. The default is 20.

    Returns
    -------
    dictionary
        keys are unique values in group_list
        values are lists of:
            0. list of lists containing topn words for each topic 
                (as encoding in per_group_relevance_dict for that topic)
            1. list of lists containing probabilities or expected counts
                for each top word in each topic 
            2. list of lists containing probabilities or expected counts
                from the overall group-specific corpus for each word in 
                each topic

    """
    groups = np.unique(group_list)
    
    
    #CODE FOR ALTERNATIVE METHOD THAT I'VE DISCARDED
    #in favor of methods that all use expected counts...
    #but which gave almost identical results
    
    # if normalized:
    #     pdist_dict = corpus_pdist_bygroup(corpus = corpus, 
    #                                       dictionary = dictionary,
    #                                       group_list = group_list)
        
    # else: #code for alternative overall count approach
    #     fdist_dict = corpus_freq_bygroup(corpus = corpus,
    #                                      dictionary = dictionary, 
    #                                      group_list = group_list)

    out_dict = {}
    for g in groups:
        order_mat = per_group_relevance_dict[g][1]
        #get top word lists for this group in terms of relevance dict's order matrix for that group
        top_word_lists = [[dictionary[i] for i in ind[:topn]] for ind in order_mat] #accessing the order matrix
        #get probabilities for each list of top words for each topic
        phi = per_group_topic_dict[g]
        top_word_topic_probs = [ [phi[i,:][j] for j in ind[:topn] ] for (i, ind) in enumerate(order_mat)] 
        
        #get overall probabiliy or overall of each word within the group
        fdist = np.sum(phi, axis = 0)
        #fdist = fdist_dict[g] #discarded method
            
        if normalized:
            #pdist = pdist_dict[g] #discarded method
            pdist = fdist/np.sum(fdist)
            top_word_overall_vals = [[pdist[i] for i in ind[:topn]] for ind in order_mat]

        else:
            top_word_overall_vals = [[fdist[i] for i in ind[:topn]] for ind in order_mat]
            
        out_dict[g] = [top_word_lists, top_word_topic_probs, top_word_overall_vals]

    return(out_dict)



def get_group_topword_vals_for_topic(topword_values_dict, topic_id):
    """
    From dictionary output by get_topword_values_dict(), 
    get the output for a single topic across all groups

    Parameters
    ----------
    topword_values_dict : output of get_topword_values_dict()
    topic_id : int, topic to get values for

    Returns
    -------
    dictionary where keys are same as topword_values_dict (the group names)
    but values are now just lists of length 3 where:
        0. is list of top words
        1. contains their corresponding probabilities/expected counts
        2. contains their corresponding corpus probabilities/counts
    """
    groups = list(topword_values_dict.keys())
    
    out_dict= {}
    for g in groups:
        out_dict[g] = [elem[topic_id] for elem in topword_values_dict[g]]
    
    return(out_dict)
    

