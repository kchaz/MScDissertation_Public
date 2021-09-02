# -*- coding: utf-8 -*-
"""

This file contains functions for processing the output of gensim LdaModels

Specifically, it contains functions for accessing the documents associated with
particular topics and/or groups


Author: Kyla Chasalow
Last edited: August 16. 2021
"""


import pandas as pd
import numpy as np
import numpy_indexed as npi
import matplotlib.pyplot as plt

import LdaOutput
import Helpers
from Helpers import figure_saver


#### Function for getting top documents associated with a topic 

def get_topic_topdoc_table(df, theta_mat, topic_id, topd, 
                               latex = False, omit_abstract = True):
    """
    Function for getting summary of top documents associated with a topic in 
    either dataframe or latex table form. 
    Includes either year, journal, and title or
    year, journal, title, and abstract

    Parameters
    ----------
    df : pandas dataframe including columns titled "Year","dc:title", "dc:description",
        and "prism:publicationName"

    theta_mat : theta matrix for a topic model as output by get_document_matrices

    topic_id : int, the topic to provide articles for

    topd : int, the number of top articles to provide
    
    latex : bool, optional
        if True, returns table in basic latex format (no captions or labels).
        The default is False.
        
        Note: latex output includes an index column - haven't yet
        found way to turn this off
        
        
    title_year_only : bool, optional
        if True, table returned includes only title and year. The default is False.

    Returns
    -------
    a pandas dataframe (if latex = False) or a string representing output of
    pandas .to_latex method
    
    
    depending on title_year_only setting, dataframe contains year, title, and 
    possibly abstract for the top topd documents associated with topic topic_id
    
    

    """
    pd.set_option('display.max_colwidth', None) #setting so that displays entire column
    
    if not omit_abstract:
        summary_df = df[["Year","dc:title","prism:publicationName","dc:description"]]
    else:
        summary_df = df[["Year","prism:publicationName","dc:title"]]
    
    #get order of theta's for each topic from best to worst
    order_mat = np.flip(np.argsort(theta_mat, axis = 1), axis =1) 
    top_ind = order_mat[topic_id][:topd]
    top_vals = np.round(theta_mat[topic_id][top_ind],2)
    
    #build final df
    out_df1 = summary_df.iloc[top_ind,:]
    out_df1.reset_index(drop=True, inplace=True)
    out_df= pd.concat([out_df1, pd.DataFrame(top_vals)], axis = 1)
    out_df.rename({0:"Theta", "dc:title":"Title", "prism:publicationName":"Journal"},inplace = True, axis = 1)
    if not omit_abstract:
        out_df.rename({"dc:description":"Abstract"}, inplace = True, axis =1)
    
    if not latex:
        return out_df
    else:
        if not omit_abstract:
            f = "c|c| p{50mm}| p{25mm} | p{50mm} | c"   #extra c for index column I'd like to remove
        else:
            f = "c|c| p{30mm} | p{85mm} | c"          #extra c for index column I'd like to remove
        out_df.reset_index(drop=True, inplace=True)
        return out_df.to_latex(column_format = f)
     

    




def get_topic_topdoc_by_journal(df, theta_mat, topic_id, topd, 
                               latex = False, omit_abstract = True):
    """
    applies get_topic_topdoc_table() to each journal in df separately
    all parameters have same meaning as in get_topic_topdoc_table()
    
    output is now a list of pandas dataframes or a list of those dataframes in 
    latex string form (if latex = True)

    """

    #get datasets for each journal    
    journals = df["prism:publicationName"]
    unique_journals = np.unique(journals)
    journal_data_list = [df[df["prism:publicationName"] == j] for j in unique_journals]

    #get theta matrix grouped by journal by first grouping indices 0...D
    theta_mat_dict = LdaOutput.get_theta_mat_by_group(theta_mat, journals)
   
    top_table_list = [ get_topic_topdoc_table(df = data,
                                              theta_mat = theta_mat_dict[journal],
                                              topic_id = topic_id,
                                              topd = topd, 
                                              latex = latex,
                                              omit_abstract = omit_abstract)
                      for journal, data in zip(unique_journals, journal_data_list)]
    
    
    return(top_table_list)






#### Look at number of topics per document

def topics_per_doc_summary(theta_mat, plot = False, bins = 10, custom_title = None,
                               color = "orange", alpha =0.5, 
                              set_figsize = True, figsize = (8,5)):
    """
    

    Parameters
    ----------
    theta_mat : numpy array as output by get_document_matrices() in LdaOutput.py
        
    plotting parameters
    -------------------
    plot : bool, optional
        If True, plots a histogram of # of non-zero topics per document
        The default is False.
        
    bins : int, number of bins to use in histogram. The default is 10.
    
    custom_title, color, alpha, set_figsize, and figsize are usual parameters for adjusting plot

    Returns
    -------
    dictionary with keys:
        
        "TopicsPerDoc" : numpy array with the number of topics that have non-zero probability
            for each document
            
        "mean" : the mean of the "TopicsPerDoc" array
        "std" : the standard deviation of the "TopicsPerDoc" array
        "median" : the median of the "TopicsPerDoc" array

    """
    non_zero_topics = np.sum(theta_mat != 0, axis = 0)
    
    out_dict = {}
    out_dict["TopicsPerDoc"] = non_zero_topics
    out_dict["mean"] = np.mean(non_zero_topics)
    out_dict["std"] = np.std(non_zero_topics)
    out_dict["median"] = np.median(non_zero_topics)
    if plot:
        if set_figsize:
            plt.figure(figsize = figsize)
        plt.hist(non_zero_topics, bins = bins, density = True,
                 color = color, alpha = alpha)
        plt.scatter(out_dict["mean"], 0, color = "red", s = 70, label = "mean", alpha = 0.8)
        plt.scatter(out_dict["median"], 0, color = "blue", s = 70, label = "median", alpha = 0.8)
        plt.legend(fontsize = 15)
        plt.xlabel("Number of Topics per Document", fontsize = 15)
        plt.ylabel("Proportion", fontsize = 15)
        if custom_title is None:
            custom_title = ""
        plt.title(custom_title, pad = 15, fontsize = 20)
        
    
    # TO DO: ADD SAVING OPTIONS
    
    return(out_dict)




def topics_per_doc_summary_by_group(theta_mat, group_list, plot = False, bins = 10,
                                     color = "orange", alpha = 0.5, figsize = (10,10)):
    
    """
    
    #TO DO: Documentation
    
    #TO DO: Turn this into a grid plotter with all grids having same y axis
    
    #TO DO: Add saving option
    
    """
    
    
    theta_mat_dict = LdaOutput.get_theta_mat_by_group(theta_mat, group_list)
    groups = np.unique(group_list)
    
    out_dict = {}
    for g in groups:
        out_dict[g] = topics_per_doc_summary(theta_mat_dict[g], 
                                                        plot = plot,
                                                        custom_title = g,
                                                        bins = bins,
                                                        color = color, 
                                                        alpha = alpha) 
        
    return(out_dict)





#look at distribution of theta values for each topic overall or by group
def theta_hist(theta_mat, topic_id, remove_zeros = False,
               title = None, xlabel = None,
               bins = 20, color = "purple",
               alpha = 0.5, density = False):
    """
    Plot histogram of theta values for topic <topic_id>. Also return
    the heights of each of the bars (e.g. for use in setting max value of all plots
    in a grid to be the same)
    
    Is used as a helper in theta_hist_by_group but can also be used on its own
    if don't want to look by group


    Parameters
    ----------
    theta_mat : numpy array as output by LdaOutput.get_document_matrices()
    topic_id : int
        topic to plot histogram of theta values for
        
    remove_zeros : bool, optional
        If True, plots only histogram of non-zero values. The default is False.
    
    bins : int, optional
        bins to use in histogram. The default is 20.
    
    for color, alpha, and density, see plt.hist documentation
    
    Returns
    -------
    the heights of the bars in the histogram

    """
    vals = theta_mat[topic_id,:]
    if remove_zeros:
        vals = vals[vals != 0]
        
    heights = plt.hist(vals, bins = bins,
            color = color, alpha = 0.5, density = density, range = (0,1))[0]
    if title is not None:
        plt.title(title, fontsize = 20, pad = 20)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize = 16)
    return(heights)




# NOTE: ISSUE - HEIGHTS ARE ONLY SET TO BE THE SAME WITHIN A SINGLE GRID

def theta_hist_by_group(theta_mat, group_list, topic_id, remove_zeros = False,
                        normalize = False, bins = 20, color = "purple", alpha = 0.5, title = None,
                        group_name = None, save_fig = False, fig_outpath = None, 
                        fig_name = "theta_hist_grid", dpi = 200,
                        fig_override = False):
    """
    Examine the distribution of topic proportion values theta by groups. Creates
    grids of at most 4 x 3 - if needed, creates multiple grids
     

    Parameters
    ----------
    theta_mat : numpy array as output by LdaOutput.get_document_matrices()
    
    group_list : list or array of group labels for each document
    
    topic_id : int, topic to examine
    
    bins...alpha are standard plt.hist() arguments

    title : str, title of plot. If none, use default involving group_name 
        argument. If not none, group_name has no role
    
    group_name : str, group name to use in title of plot as in    
        <<Distribution of Documents' Topic <topic_id> Proportions by <group_name>">>

    save_fig...fig_override are the standard figure saving options. See 
        fig_saver() function in Helpers.py

   
    Returns
    -------
    None.

    """
    assert theta_mat.shape[1] == len(group_list), "Theta matrix and group_list dimension mismatch"
    plt.rcParams.update({'font.family':'serif'})
    
    
    if group_name is None:
        group_name = "Group"
    
    #split up theta matrix by group
    theta_mat_dict = LdaOutput.get_theta_mat_by_group(theta_mat, group_list)
    
    #get info about groups
    group_names = list(theta_mat_dict.keys())    
    num_groups = len(group_names)
    
    #figure out how many grids to create
    grid_list = Helpers.get_grids(to_plot_list = list(range(num_groups)),
                                  num_col = 3,
                                  num_row = 4)
    
    
    
    #create grids
    for j, plot_list in enumerate(grid_list):
                
        #figure out grid size
        if len(plot_list) <= 2:
            adjust = True
        else:
            adjust = False
            
        nrows, ncols = Helpers.figure_out_grid_size(num_plots = len(plot_list), 
                                                    num_cols = 3, max_rows = 4,
                         adjust_for_low_num_plots = adjust)
        
        #grid size specific settings
        if nrows == 1:
            top = .73
        elif nrows == 2:
            top = 0.85
        elif nrows == 3:
            top = 0.90
        elif nrows == 4:
            top = 0.90
        
        #set up grid
        figsize = (ncols * 5, nrows * 6)
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=figsize)
        fig.text(0.5, -.07,  r"Document Topic Proportion ($\theta_{d%d}$)"%topic_id, ha='center', fontsize = 22)
        if normalize:
            ylab = "Density"
        else:
            ylab = "Document Count"
        fig.text(-.05, 0.4, ylab, va='center', rotation='vertical', fontsize = 22)
        fig.tight_layout(h_pad=10, w_pad = 6)

        #holders
        ax_list = [] #collect the axes
        counts_list = [] #collect the heights of all the histogram bars
        
        #main plotting
        num_plots = len(plot_list)
        for i in range(nrows * ncols):
            ax = plt.subplot(nrows, ncols, i+1)
            ax_list.append(ax)
            
            if i < num_plots:
                theta_mat = theta_mat_dict[group_names[plot_list[i]]]
                counts_list.append(theta_hist(theta_mat = theta_mat,
                                              topic_id = topic_id,
                                              remove_zeros = remove_zeros,
                                              bins = bins, 
                                              color = color,
                                              alpha = alpha, 
                                              density = normalize))
                plt.title(group_names[plot_list[i]], pad = 20, fontsize = 20)
            #turning off extra axes
            else:
                ax.set_visible(False)

        #set axis limits to make all plots comparable
        max_val = np.max([np.max(elem) for elem in counts_list])
        for ax in ax_list:
            ax.set_ylim((0,max_val*1.1))
            ax.set_xlim(0,1)

        #default title
        if title is None:
            title = "Distribution of Documents' Topic %d Proportions by %s" % (topic_id, group_name)
            if remove_zeros:
                title += "\n(zero-truncated, normalized per group)"
            else:
                title += "\n(normalized per group)"
        plt.suptitle(title, fontsize = 28)
        plt.subplots_adjust(top= top)     
        
        
        #saving each grid
        if save_fig:
                figure_saver(fig_name = fig_name + str(j), #number the grids
                         outpath = fig_outpath,
                         dpi = dpi,
                         fig_override = fig_override,
                         bbox_inches = "tight")













### DOCUMENT TRAJECTORY PLOT

#FUTURE EXTENSIONS
#to do: consider also returning actual theta values and adding thresholding
#capcity so can also plot, say, only documents with max theta over a certain value
#for given topic_id in root_k
#find way to plot all documents ever affiliated with the label 7, say


def get_max_topic_matrix(theta_dict, min_K = None, max_K = None):
    """
    Get a matrix with the ID of the topic with the largest theta value for each document
    in each k-topic model represented in theta_dict
    

    Parameters
    ----------
    theta_dict : dictionary of numpy arrays output by LdaOutput.get_document_matrices
        keys of document should be corresponding k values (ints)
    min_K nad max_K:
        optionally don't look at all k in theta_dict but only those in [min_K, max_K]

    Returns
    -------
    Given R possible k-values considered (either all keys of theta_dict or those within
    specified range), matrix is R x D where D is number of documents. The (r,d)^th 
    entry is the topic ID of the topic which has the maximum $\theta$ value for document $d$
    in the r-topic model.

    """
    Kvals = list(theta_dict.keys())
    if max_K is not None:
        Kvals = [k for k in Kvals if k <= max_K]
    if min_K is not None:
        Kvals = [k for k in Kvals if k >= min_K]
        
    D = theta_dict[Kvals[0]].shape[1]
    out_array1 = np.zeros((len(Kvals),D), dtype = "int64")
    #out_array2 = np.zeros((len(Kvals),D))
    
    for i,k in enumerate(Kvals):
        theta_mat = theta_dict[k]
        out_array1[i,:] = np.argmax(theta_mat, axis = 0) 
        #out_array2[i,:] = np.max(theta_mat, axis = 0)     
        
    return(out_array1, Kvals)
    

def _get_max_widths_per_k(max_topic_matrix):
    """Get the largest number of document assigned to some topic ID for each k
     use this to help with spacing of documents in plot for each k"""
    out_array = np.zeros(max_topic_matrix.shape[0])
    for i, row in enumerate(max_topic_matrix):
        out_array[i] = np.max(np.bincount(row))
    return(out_array)



def plot_doc_trajectories(theta_dict, root_k, topic_id = None, plot_all = False, 
                          min_K = None, max_K = None,
                         single_color = "purple", cmap = "tab20",
                         figsize = (15,20), shift_width = True,
                         save_fig = False, fig_outpath = None, 
                         fig_name = "doc_trajectories", dpi = 200,
                         fig_override = False):
    """
    First, define a document's topic 'affiliation' here to mean the topic for which it
    has the highest theta value. This is a bit crude, as some documents will have a 
    substantial mix of multiple topics. In a sense, I'm ignoring the soft clustering 
    of LDA here to assign hard clusters.

    The idea here is to plot the trajectories of multiple documents across models in terms of 
    their topic 'affiliations'. While there is no requirement that topic IDs correspond 
    to the same topic across models, if models are trained with the same random state
    I have found there is a lot of consistency. 
    
    Diagonal lines in the plots produced can signify two things.
    If a large number of documents affiliated with topic 4 (say) in model k become
    affiliated with topic 8 in model k+5, then perhaps there has been a label switch
    and topics 4 and 8 are similar. If there are only a few breakaways of documents 
    from topic 4 to other topic IDs, then this may reflet topics splitting, 
    new topics arising that better describe certain documents or
    theta values slightly shifting to change the relative size rankings of a document's
    component topics.

    Parameters
    ----------
    theta_dict : dictionary of numpy arrays output by LdaOutput.get_document_matrices
        keys of document should be corresponding k values (ints)
    
    root_k : int
        if plot_all is True, this only specified the model to use for coloring the documents
        All documents affiliated with topic i in the root_k model will have same color
        
        if plot_all is False, this helps specify which documents to plot the trajectories for.
        Namely, the plot shows only documents affiliated with topic topic_id in the 
        root_k-topic model
        
    topic_id : int, optional
        if plot_all is False, then this specifies which topic to examine. See description of
        root_k parameter. The default is None.
    
    min_K nad max_K:
        optionally don't look at all k in theta_dict but only those in [min_K, max_K]
    
    plot_all : bool, optional
        if True, plots ALL document trajectories. Note that for large D (already when in the 1000's)
        this becomes a very hard to read, possibly meaningless plot. Note that documents are colored
        by their topic affiliation in the root_k model
        
        When False, requires that topic_id is specified and plots only documents affiliated with
        topic <topic_id> in the <root_k>-topic model.
        
        The default is False.
        
    single_color : str, optional
        when plot_all is False, this specifies color of plot.
        The default is "purple".
    
    cmap : name of a color map, optional
        used with plot_all = True. The default is "tab20".
        Warning: default tab20 will not work as well for root_k > 20
        and using plot_all = True with root_k > 20 is not advised since
        it will become very hard to distinguish colors
        
    figsize : tuple, optional. The default is (15,20).
    
    shift_width : bool, optional
        if True, the function will shift each document line slightly in an 
        attempt to avoid them all being plotted on top of each other.
        The amount of shift is consistent within the plot and as you change
        topic_id and root_k, for it is is determined as 1 divided by
        the maximum number of documents affiliated with a single topic for any K
        
        see _get_max_widths_per_k() helper function
        
        The default is True.

    save_fig...fig_override are the standard figure saving options. See 
        fig_saver() function in Helpers.py


    Returns
    -------
    None.

    """
    
    ID_matrix, Kvals = get_max_topic_matrix(theta_dict, min_K = min_K, max_K = max_K)
    D = theta_dict[Kvals[0]].shape[1]
    
    bin_widths = _get_max_widths_per_k(ID_matrix)
    
    #divide documents by their topic ID affiliation for the root_k-topic model
    #this will determine coloring if plot_all = True
    ind = Kvals.index(root_k)
    labels, doc_groups = npi.group_by(keys = ID_matrix[ind,:], values = list(range(D)))

    #get colors from some color map
    colors = plt.get_cmap(cmap)(range(root_k))

    #overall plot settings
    plt.figure(figsize = figsize)
    plt.xlabel("Number of Topics in Model", fontsize = 18)
    plt.ylabel(r"Topic ID corresponding to maximum $\theta$ value for each document", fontsize = 18)
    plt.ylim(0, max(Kvals))
    plt.yticks(ticks = list(range(max(Kvals))), labels = list(range(max(Kvals))), fontsize = 15)
    plt.xticks(ticks = Kvals, labels = Kvals, fontsize = 15)

    #plot in groups of documents determined by root_k
    plt.rcParams.update({'font.family':'serif'})
              
    #plot just the documents affiliated with topic <topic_id> in the <root_k> topic model
    if not plot_all:
        title = "Document Trajectories of documents for which \n Topic %d has greatest $\\theta_{dk}$ value " % topic_id
        title += "in %d-Topic Model" % root_k
        assert topic_id is not None, "topic_id must not be None if plot_all = False"
        assert topic_id < root_k, "invalid topic_id value given root_k value"
        try:
            l = list(labels).index(topic_id)  #TO DO: possible for it not to be in list - address that edge case
        except:
            l = None
        if l is not None:
            doc_ids = doc_groups[l]
            for j, doc in enumerate(doc_ids):
                y = ID_matrix[:,doc]
                if shift_width:
                    y = y + 1/np.max(bin_widths)* j #width adjustment
                plt.scatter(Kvals, y, color = single_color, alpha = 0.15)
                plt.plot(Kvals, y, color = single_color, alpha = 0.15, linewidth = 1)
        else:
            print("Cannot plot: no documents have topic_id %d as their max theta topic in the %d-Topic model" % (topic_id, root_k))

    #plot all documents' trajectories (WARNING: THIS PLOT GETS VERY MESSY)
    else:
        title = "Document Trajectories \n (rooted at K = %d)" % root_k
        for j, topic_id in enumerate(labels):
            doc_ids = doc_groups[j]
            for doc in doc_ids:
                y = ID_matrix[:,doc]
                if shift_width:
                    y = y + 1/np.max(bin_widths)* j #width adjustment - each doc shifted slightly up
                plt.scatter(Kvals, y, color = colors[j], alpha = 0.1)
                plt.plot(Kvals, y, color = colors[j], alpha = 0.2, linewidth = 0.5)

    plt.title(title, fontsize = 25, pad = 20)


    # FIGURE SAVING OPTIONS    
    if save_fig:
                figure_saver(fig_name = fig_name,
                         outpath = fig_outpath,
                         dpi = dpi,
                         fig_override = fig_override,
                         bbox_inches = "tight")