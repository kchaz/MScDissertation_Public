# -*- coding: utf-8 -*-
"""

This file contains various functions meant to process data files for the
Evolution of the Social Sciences Project.  The data are assumed to come from
Scopus but the functions are hopefully generally applicable to such data.


Author: Kyla Chasalow
Last edited: September 2, 2021


""" 
import pandas as pd
import numpy as np
import numpy_indexed as npi
import matplotlib.pyplot as plt
import langid #language detection
import nltk
import os

import Helpers 
from Helpers import filename_resolver
from Helpers import figure_saver





### Summarize missingness

def missing_summary(df, save = False, 
                    outpath = None,
                    filename = "missing_summary",
                    override = False):
    """
    
    Parameters
    ----------
    df : a pandas dataframe with n rows and r columns
    
    save options:
        if save = True, saves dataframe as csv file. If outpath specified,
        saves it there. else in working directory. If override = True, will use filename.csv
        as filename and override any existing files of same name. If override = False,
        will resolve file name by trying _1, _2, _3 etc. until finds name that does not
        already exist.
        
    Returns
    -------
    a pandas dataframe containing r rows and 3 columns
        col 0: variable name
        col 1: # observations missing for that variable
        col 2: % observations missing for that variable

    """
    n = df.shape[0]
    missing_counts = [df.loc[:,elem].isnull().sum() for elem in df.keys()]
    missing_percents = np.round(np.array(missing_counts)/n,4)
    missing_df = pd.DataFrame({"variable":list(df.keys()),
                               "counts":missing_counts,
                               "percents":missing_percents})
    
    if save:
        if not override:
            filename = filename_resolver(filename = filename, 
                                         extension = "csv", 
                                         outpath = outpath)   
        if outpath is not None:
            filename = os.path.join(outpath, filename + ".csv")
        else:
            filename = filename + ".csv"
        
        missing_df.to_csv(filename, index = False)
    
    return(missing_df)
    



def missing_abstract_summary(df,
                             save = False, 
                             outpath = None,
                             filename = "missing_abstract_summary",
                             override = False):
    """
    
    Function to breakdown abstract missingness by type of observation
    (Article, Erratum, Editorial etc.). Also returns a count of how many
    of the articles with missing abstracts contain a suggestive title like
    'erratum,' which usually corresponds to there actually being no abstract

    Parameters
    ----------
    df : dataframe
        must contain a column called "dc:description" and a column called
        "subtypeDescription"
        
    save options:
        if save = True, saves dataframe (0th object returned) as csv file. If outpath specified,
        saves it there. else in working directory. If override = True, will use filename.csv
        as filename and override any existing files of same name. If override = False,
        will resolve file name by trying _1, _2, _3 etc. until finds name that does not
        already exist.

    Returns
    -------
    tuple containing
    
    0. pandas dataframe with row for each type of abstract and 3 columns:
        0. number of observations of each type
        1. number missing abstracts for each type
        2. percent missing abstracts for each type
        
    1. count of how many of the missing article titles contain the word "erratum", "response"
    "comment","addundum" "reply," or "correction" 
    
    
    None: if some subtypes exist only in full dataset but not among observations
    with missing abstracts, these will be recored as NaN for num_missing and percent_missing

    """
    #get missingness summaries by type
    missing_abstracts_df = df[df["dc:description"].isnull()]
    missing_types = pd.DataFrame(missing_abstracts_df.subtypeDescription.value_counts())
    overall_types = pd.DataFrame(df.subtypeDescription.value_counts())
    frac_missing = missing_types/overall_types
    frac_missing.rename(columns = {"subtypeDescription":"percent_missing"}, inplace = True)
    
    #create dataframe
    overall_types.rename(columns = {"subtypeDescription":"total_counts"}, inplace = True)
    missing_types.rename(columns = {"subtypeDescription":"num_missing"}, inplace = True)
    
    summary = pd.concat([overall_types, missing_types, frac_missing], axis = 1)
    
    #TO DO: REPLACE WITH REGEX APPROACH
    #check whether any of the missing articles have below target words in them
    missing_articles = missing_abstracts_df[missing_abstracts_df.subtypeDescription == "Article"]
    target_words = ["erratum","comment","response","addendum","correction", "reply"]
    titles = missing_articles.loc[:,"dc:title"]
    count = 0
    for i in range(0,len(missing_articles)):    
        if any(item.lower() in target_words for item in nltk.word_tokenize(titles.iloc[i])):
            count += 1
            
    if save:
        if not override:
            filename = filename_resolver(filename = filename, 
                                         extension = "csv", 
                                         outpath = outpath)   
        if outpath is not None:
            filename = os.path.join(outpath, filename + ".csv")
        else:
            filename = filename + ".csv"
        
        summary.to_csv(filename, index = False)
    
    
    return(summary, count)







### Extract Year from prism:coverDate column


def extract_prism_year(df):
    """
    
    Parameters
    ----------
    df : pandas data frame
        must contain a "prism:coverDate" column

    Returns
    -------
    list of same length as df containing year for each row of df
    as obtained from prism:coverDate column. Years are of type int.

    """
    prism_years = df.loc[:,"prism:coverDate"]
    out_years = [int(entry[0:4]) for entry in prism_years]
    return(out_years)








### ANALYZE YEARS

def year_breakdown(array, plot = False, title = "Number of Observations per Year", 
                   figsize = (15,4), set_figsize = True,
                   set_xlab = True, set_ylab = True,
                   y_upper_lim = None,
                   min_plot_year = None, max_plot_year = None,
                   save_fig = False, fig_outpath = None, 
                   fig_name = "year_breakdown", dpi = 200,
                   fig_override = False):
    """
    0. Counts the number of missing years
    1. Checks whether all years are present in the range of years
    provided in array, ignoring nan values
    2. Provides minimum and maximum year included in array
    3. Provides count of # of observations for each year included in the range
    4. Optionally, plots barplot of number of observations per year 
    
    Parameters
    ----------
    array : np.array
        np.array of possibly non-continguous years as outputted by
        extract_all_years() function
    
    plot: optionally plot missingness by year
    
    title: str, default "Number of Observations per Year"

    figsize: tuple
        optionally adjust figsize if plotting
        
    set_figsize : bool
        option to not set figure size, of use for gridding
        
    set_xlab and set_ylab: bools
        option to not set x or y labels or both
        
    y_upper_lim : int or float
        optionally set upper limit of y axis on plot
        
    min_plot_year and max_plot_year can be optionally specified to
    change the year range over which bar plot of obs by year is plot. Will not
    affect minimum and maximum year output by funciton
        
    save options:
        if save = True, saves dataframe (0th object returned) as csv file. If outpath specified,
        saves it there. else in working directory. If override = True, will use filename.csv
        as filename and override any existing files of same name. If override = False,
        will resolve file name by trying _1, _2, _3 etc. until finds name that does not
        already exist.

    Returns
    -------
    dictionary with the following keys and values
    
    "num_missing" : number of missing values contained in array (int)
    "min_year" : the earliest year in array (int)
    "max_year" : the latest year in array (int)
    "year_counts" : np.array of lenth max_year-min_year + 1 with count for every year (including 0 
               for any unobserved years in the range min_year:max_year 
    "year_gaps" : a list of tuples where each tuple represents a gap in years observed. The values in
                the tuple ARE years observed. For example, if array was [2001,2002,2003,2006]
                then tuple representing the absence of 2004 or 2005 would be (2003,2006).
                If there are no gaps, this is []
    "earliest_continuous_year" : the first year for which all years after and including that year
        have at least one observation (occur at least once in array)
    """
    #number of missing years
    num_missing = sum([1 for i in array if str(i) == 'nan'])
    
    #remove nan values and make sure have ints
    cleanedYears = np.array([int(i) for i in array if str(i) != 'nan'])
    #obtain unique years included
    unique_years = np.unique(cleanedYears)
   
    #earliest and latest year in array
    min_year = unique_years.min()
    max_year = unique_years.max()
    
    
    #check whether continuous and find the break points and earliest year after
    #which observations ARE continuous
    gaps = [] #hold each point where continuous switches from True to False
    for i, elem in enumerate(unique_years[:-1]):
        if elem + 1 not in unique_years:
            gaps.append((unique_years[i],unique_years[i+1]))

    if gaps == []:
        earliest_continuous_year = min_year
    else:
        earliest_continuous_year = gaps[-1][1] #the last gap
            
    
    #min/max slicing necessary or it starts counting 0,1,2...2021
    #with 0 counts up to before earliest year. If there are 0 in range 
    #of years observed, these will be included as 0's
    year_counts = np.bincount(cleanedYears)[min_year:max_year+1]
    
    if min_plot_year is None:
        min_plot_year = min_year
    if max_plot_year is None:
        max_plot_year = max_year
    
    if plot:
        plt.rcParams.update({'font.family':'serif'})
        if set_figsize:
            plt.figure(figsize = figsize)
        plt.bar(np.arange(min_year, max_year+1), year_counts, color = "green", alpha = 0.5)
        plt.title(title, 
                  fontsize = 22, pad = 15)
        if set_xlab:
            plt.xlabel("Year", fontsize = 18)
        if set_ylab:
            plt.ylabel("Counts", fontsize = 18)
        plt.yticks(fontsize = 16)
        plt.xticks(fontsize = 16)
        plt.xlim(min_plot_year -1, max_plot_year + 1)
        if y_upper_lim is not None:
            plt.ylim(0, y_upper_lim)
        if save_fig:
               figure_saver(fig_name = fig_name, 
                         outpath = fig_outpath,
                         dpi = dpi,
                         fig_override = fig_override,
                         bbox_inches = "tight"
                        )   
    
    out_dict = {}
    out_dict["num_missing"] = num_missing
    out_dict["min_year"] = min_year
    out_dict["max_year"] = max_year
    out_dict["year_counts"] = year_counts
    out_dict["year_gaps"] = gaps
    out_dict["earliest_continuous_year"] = earliest_continuous_year
        
    return(out_dict)




### ANALYZE ABSTRACTS
def abstract_breakdown(df, plot = False, title = "Percent of Observations Missing Abstracts",
                       figsize = (15,4), 
                       set_figsize = True, set_xlab = True, set_ylab = True,
                       min_plot_year = None, max_plot_year = None,
                       save_fig = False, fig_outpath = None, 
                       fig_name = "abstract_year_breakdown", dpi = 200,
                       fig_override = False):
    """

    1. Counts how many abstracts are missing for each year
    2. Calculates percentage of observations per year missing abstracts
    3. Optionally, plots percent missing by year
    
    Parameters
    ----------
    df: pandas dataframe
        must contain a "Year" column and a "dc:description" column
        
        
    plot: bool
        if True, will plot bar plot of abstracts missing by year

    title: string
         Title of plot, default "Percent of Observations Missing Abstracts"
         
    set... options : bools, optionally turn off various aspects of plot
    
    min_plot_year and max_plot_year can be optionally specified to
    change the year range over which bar plot of obs by year is plot. 
    
    save options:
            if save = True, saves dataframe (0th object returned) as csv file. If outpath specified,
            saves it there. else in working directory. If override = True, will use filename.csv
            as filename and override any existing files of same name. If override = False,
            will resolve file name by trying _1, _2, _3 etc. until finds name that does not
            already exist.

    Returns
    -------
    tuple containing
        0. dictionary containing years as keys and count of abstracts missing
        in each year as values
        1. dictionary containing years as keys and percent of observations for 
        each year missing abstract as values
        
        *Note: for both dictionaries, if there are 0 observations in a year
        within range of minimum and maximum year, that year will be in the dictionary 
        but its value will be np.nan.
        However, when plotting, these will be treated as 0's
    
    """
    assert "Year" in df.keys(), "df must contain a column called 'Year'"
    assert "dc:description" in df.keys(),"df must contain a column called dc:description"
    
    #remove any missing years
    cleanedYears = df.Year.dropna().astype(int)
    
    #obtain minimum and maximum years in data frame
    min_year = cleanedYears.min()
    max_year = cleanedYears.max()
    #indexing necessary because count will start at year 0 otherwise...
    year_counts = np.bincount(cleanedYears)[min_year:max_year+1]
    
    #obtain indices of missing abstracts and select those rows of df
    null_data = df[df.loc[:,"dc:description"].isnull()]
    num_null = null_data.shape[0]
    
    #create dictionary with years as keys 
    year_dict = {}
    for i in range(min_year, max_year+1):  #min and max year defined above
        year_dict[i] = 0
        year_dict["nan"] = 0
        
    #figure out which column index is year index
    ind = np.where(df.keys() == "Year")[0][0]
    
    #fill dictionary with number missing as values
    for i in range(0, num_null):
        if np.isnan(null_data.iloc[i,ind]):
            year_dict["nan"] += 1
        else:
            year_dict[int(null_data.iloc[i,ind])] += 1
    
    #set year_dict values to np.nan if there are no obs for that year
    years = list(range(min_year, max_year+1))
    for i,c in enumerate(year_counts):
        if c == 0:
            year_dict[years[i]] = np.nan
    
    #calculate percent missing for each year
    year_percents = {}
    for elem in year_dict:
        if elem == "nan":
            pass    
        else:
            elem = int(elem) 
            num_obs = year_counts[elem - min_year] 
            #avoid division by 0 
            if num_obs != 0:
                year_percents[elem] = year_dict[elem]/num_obs
            else:
                year_percents[elem] = np.nan
    
      
          
    if plot:
           
        if min_plot_year is None:
            min_plot_year = min_year
        if max_plot_year is None:
            max_plot_year = max_year
     
        
        #for plotting only, replace any np.nan values with 0 
        vals = list(year_percents.values())
        vals = np.nan_to_num(vals, copy=True, nan=0, posinf=None, neginf=None)
        
        if set_figsize:
            plt.figure(figsize = figsize)
        plt.rcParams.update({'font.family':'serif'})
        plt.bar(np.arange(min_year, max_year+1), 
                vals,
                alpha = .75)
        if set_xlab:
            plt.xlabel("Year", fontsize = 18)
        if set_ylab:
            plt.ylabel("Percent Missing", fontsize = 18)
        plt.title(title, 
                  fontsize = 22, pad = 15)
        plt.yticks(fontsize = 16)
        plt.xticks(fontsize = 16)
        plt.xlim(min_plot_year -1, max_plot_year + 1)
        plt.ylim(0,1)

        if save_fig:
               figure_saver(fig_name = fig_name, 
                         outpath = fig_outpath,
                         dpi = dpi,
                         fig_override = fig_override,
                         bbox_inches = "tight"
                        )   
    
    return(year_dict, year_percents)








def abstract_length_breakdown(df, plot = False, title = None, color = "lightblue", 
                              alpha = 1, figsize = (15,4), 
                              set_figsize = True, set_xlab = True, set_ylab = True, 
                              y_upper_lim = None, 
                              save_fig = False, fig_outpath = None, 
                              fig_name = "abstract_length_breakdown", dpi = 200,
                              fig_override = False):
    """

    Drops rows with null abstracts and null years, for remaining, calculates 
    abstract lengths and provides summary statistics overall and by year
    
    NOTE: Length is measured here in characters since abstracts in df not yet
    split into words

    Parameters
    ----------
    df : pandas data frame
        must contain a "dc:description" and a "Year" column
    plot : Bool, optional
        if true, plots boxplots of abstract length by year 
    title : str, optional
        optionally add on to default "Abstract Length by Year" title

    color, alpha, figsize are plotting parameters

    parameters that begin with "set" turn on and off various plot elements

    Returns
    -------
    dictionary containing various summary information about abstracts
    
    length_type : "characters"
    overall_mean : overall mean length
    overall_std : overall standard deviation of lengths
    avg_by_year : length average by year
    std_by_year : length std by year
    abstract_lengths : length of each abstract
    grouped_lengths : lengths grouped by year

    """
    #drop null abstracts or rows without a year
    abstracts = df[["dc:description","Year"]].dropna() 
    
    
    # if min_year is None:
    #         min_year = np.min(abstracts.Year)
    # if max_year is None:
    #         max_year = np.min(abstracts.Year)
        
    # abstracts = abstracts[abstracts.Year >= min_year and abstracts.Year <= max_year]
        
    #vectorize length function
    vec_len = np.vectorize(len)
    abstract_lengths = vec_len(abstracts["dc:description"])
    grouped_lengths = npi.group_by(abstracts.Year).split(abstract_lengths)
         
    overall_mean = np.mean(abstract_lengths)
    overall_std = np.std(abstract_lengths)
    
    #averages by year
    avg_by_year = npi.group_by(abstracts.Year).mean(abstract_lengths)
    std_by_year = npi.group_by(abstracts.Year).std(abstract_lengths)
    
    #year_counts = np.bincount(abstracts.Year)[np.min(abstracts.Year):]

    summary_dict = {}
    summary_dict["length_type"] = "characters"
    summary_dict["overall_mean"] = overall_mean
    summary_dict["overall_std"] = overall_std
    summary_dict["avg_by_year"] = avg_by_year
    summary_dict["std_by_year"] = std_by_year
    summary_dict["abstract_lengths"] = abstract_lengths
    summary_dict["grouped_lengths"] = grouped_lengths

    
    if plot:
        
        plt.rcParams.update({'font.family':'serif'})
        if set_figsize:
            plt.figure(figsize = figsize)

        #needed for box plot
        labels = np.unique(abstracts.Year)
        
        plt.boxplot(grouped_lengths, vert=True, labels=labels, 
                            patch_artist = True,
                            boxprops=dict(facecolor=color, color=color, alpha = alpha),
                            medianprops=dict(color= "black")) 
        if title is None:
            title = "Abstract Lengths by Year"
        
        ax = plt.gca()
        ax.set_xticklabels(labels=labels,rotation=90)
        plt.title(title, fontsize = 20, pad = 15)
        if set_xlab:
            plt.xlabel("Year", fontsize = 18)
        if set_ylab:
            plt.ylabel("Length (in characters)", fontsize = 18)
        plt.yticks(fontsize = 13)
        plt.xticks(fontsize = 13)
        if y_upper_lim is not None:
            plt.ylim(0,y_upper_lim)
        else:
            plt.ylim(0,)

        if save_fig:
               figure_saver(fig_name = fig_name, 
                         outpath = fig_outpath,
                         dpi = dpi,
                         fig_override = fig_override,
                         bbox_inches = "tight"
                        )   
        
    return(summary_dict)
    





### Language

## found in test runs that quite accurate for 300 characters
## so making that default. More accurate the more characters you use

def abstract_languages(df, thresh = 300, verbose = False):
    """
    
    Runs language detector on every abstract in df
    
    Parameters
    ----------
    df: pandas dataframe
        must contain a "description" column
    
    thresh: int
        function will use first thresh characters of abstract
        to classify. If shorter than thresh, uses all of abstract
    
    verbose: if True, will print iteration number very
    1000 iterations so can track progress

    Returns
    -------
    list containing language output by langid language detector
    contains "NA" if abstract is missing
    contains "error" if langid encountered some error in detection
    
    """
    

    n = df.shape[0]
    
    abstracts = df.loc[:,"dc:description"]
    abs_lang = [None] * n
    for i in range(0,n):
        abstract = abstracts.iloc[i]
        if verbose and i % 1000 == 0:
            print(i)
            
        if str(abstract) == "nan":
            abs_lang[i] = None
        else:
            try:
                abstract = abstract[0:thresh]
                lang = langid.classify(abstract)[0]
            except:
                lang = "error"
            abs_lang[i] = lang

    return(abs_lang)




def lang_breakdown(langs):
    """
    
    Parameters
    ----------
    langs : list or np array containing language codes 
    written to process output of abstract_languages() function

    Returns
    -------
    tuple containing
        0. number of english abstracts
        1. number of non-english abstracts
        2. number of NA abstracts
        3. list of english indices
        4. list of non-english indices
        5. list of NA indices
        6  bool for whether there are any "error" entries

    """
    n = len(langs)
    
    en_list = []
    na_list = []
    not_en_list = []
    error_count = 0

    for j in range(0,n):
        if langs[j] == "en":
            en_list.append(j)
        elif langs[j] is None:
            na_list.append(j)
        elif langs[j] == "error":
            error_count +=1
        else:
            not_en_list.append(j)
            
    num_en = len(en_list)
    num_na = len(na_list)
    num_not_en = len(not_en_list)
    
    return(num_en, num_not_en, num_na, 
           en_list, not_en_list, na_list,
           error_count)







#### Wrappers for creating plots above for multiple journals in a grid



def by_year_grid_plotter(journal_titles, datasets, max_val, 
                                plottype, suptitle = None,
                                save_fig = False, fig_outpath = None, 
                                fig_name = None, dpi = 200,
                                fig_override = False):
    """
    plot count by year or % missing by year plots or abstract lengths in characters by year
    as created by year_breakdown(), abstract_breakdown(), and abstract_length_breakdown()
    in a grid of 5 x 1
    
    Helper function for  by_year_grid()
    
    Warning: the length plot will not per se be aligned by year across the grid. See
    Groupbox script for a better approach to comparing lengths by year
    

    Parameters
    ----------
    journal_titles : list of strings with journal names
    
    datasets : corresponding list of pandas data frames for each journal
    
    max_val : int, maximum year count for any journal
   
    plottype : str, one of "counts","missing", or"character_lengths"- determines which kind of
        plots function creates
        
    save...fig_override are the standard figure saving options. See 
        fig_saver() function in Helpers.py
        
        if fig_name is none, name is plottype_grid


    Returns
    -------
    None.

    """
    options = ["counts","missing","character_lengths"]
    assert plottype in options, "plottype must be one of " + str(options)
    if fig_name is None:
        fig_name = "%s_grid" % plottype
        
    #overall plotting parameter
    plt.rcParams.update({'font.family':'serif'})
    
    
    #Get max and min overall years
    max_overall_year = np.max([np.max(data.Year) for data in datasets])
    min_overall_year = np.min([np.min(data.Year) for data in datasets])

    
    nrows, ncols = Helpers.figure_out_grid_size(num_plots = len(journal_titles),
                                                num_cols = 1,
                                                max_rows = 5,
                                                adjust_for_low_num_plots = False)
    
    figsize = (15, nrows * 4)
    if nrows == 1:
        top = 0.70
    elif nrows == 2:
        top = 0.80
    elif nrows == 3:
        top = 0.85
    elif nrows == 4:
        top = 0.88
    elif nrows == 5:
        top = 0.90
    
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=False, figsize=figsize)
    fig.text(0.5, -.07,  "Year", ha='center', fontsize = 25)
    fig.tight_layout(h_pad=8, w_pad = 6)

    if plottype == 'counts':
        fig.text(-.05, 0.4, "Counts", va='center', rotation='vertical', fontsize = 25)
    
        for i, journal in enumerate(journal_titles):
            plt.subplot(nrows,ncols,i+1)
            year_breakdown(datasets[i].Year, 
                                    plot = True, set_figsize = False, 
                                    set_ylab = False, set_xlab = False, y_upper_lim = max_val *1.1,
                                    min_plot_year = min_overall_year,
                                    max_plot_year = max_overall_year,                  
                                    title = journal,
                                    save_fig = False,
                                    )
        if suptitle is None:
            suptitle = "Number of Observations per Year"
        plt.suptitle(suptitle, fontsize = 30)
        plt.subplots_adjust(top= top)     

    
    #-----------------------------------------------------------
    elif plottype == 'missing':
        fig.text(-.05, 0.4, "Percent missing", va='center', rotation='vertical', fontsize = 25)
    
        
    
        for i, journal in enumerate(journal_titles):
            plt.subplot(nrows,ncols,i+1)
            year_dict, year_percents = abstract_breakdown(datasets[i], 
                                                          plot = True, set_figsize = False,
                                                          set_ylab = False, set_xlab = False,
                                                          title = journal,
                                                          min_plot_year = min_overall_year,
                                                          max_plot_year = max_overall_year
                                                           )
        if suptitle is None:
            suptitle = "Percent of Observations Missing Abstracts"
        plt.suptitle(suptitle, fontsize = 30)
        plt.subplots_adjust(top= top)     

    #-----------------------------------------------------------
    elif plottype == 'character_lengths':
        fig.text(-.09, 0.4, "Abstract length \n (in characters)", va='center', rotation='vertical', fontsize = 25)
        for i, journal in enumerate(journal_titles):
                plt.subplot(nrows,ncols,i+1)
                abstract_length_breakdown(datasets[i], 
                                          plot = True,
                                          title = None,
                                          color = "lightblue", 
                                          alpha = 1,
                                          set_figsize = False, 
                                          set_xlab = False, 
                                          set_ylab = False,
                                          y_upper_lim = None)
    
        if suptitle is None:
            suptitle = "Abstract Length by Year"
        plt.suptitle(suptitle, fontsize = 30)
        plt.subplots_adjust(top= top)     


    #Figure saving options here
    if save_fig:
        figure_saver(fig_name = fig_name, 
                 outpath = fig_outpath,
                 dpi = dpi,
                 fig_override = fig_override,
                 bbox_inches = "tight")

    
    
def by_year_grid(df, plottype, suptitle = None,
                 save_fig = False, fig_outpath = None, 
                                fig_name = None, dpi = 200,
                                fig_override = False):
    """
    by_year summary plots for all journals in df separately, in grids of up to 5 x 1

    either counts or % missing by year
    

    Parameters
    ----------
    df : pandas dataframe from SCOPUS data. Must contain a "Year" column added on
    plottype : str, one of "counts" or "missing"
        DESCRIPTION.
    save...fig_override are the standard figure saving options. See 
        fig_saver() function in Helpers.py
        
        if fig_name is none, name is plottype_grid

        in any case, fig_name gets appended by # to mark each different grid
    
    Returns
    -------
    None.

    """
    if fig_name is None:
        fig_name = "%s_grid" % plottype
   
    
    
    journal_titles = np.unique(df["prism:publicationName"])
    datasets = [df[df["prism:publicationName"] == journal] for journal in journal_titles]
    max_val = np.max([np.max(data.Year.value_counts()) for data in datasets])
    
    grid_list = Helpers.get_num_grids(to_plot_list = list(range(len(journal_titles))),
                                      num_col = 1,
                                      num_row = 5)    
    for i, grid in enumerate(grid_list):
        titles = journal_titles[grid] 
        data = [datasets[i] for i in grid] 
        by_year_grid_plotter(journal_titles = titles, 
                                    datasets = data,
                                    max_val = max_val,
                                    suptitle = suptitle,
                                    plottype = plottype,
                                    save_fig = save_fig,
                                    fig_name = fig_name + str(i),
                                    fig_outpath = fig_outpath,
                                    dpi = dpi,
                                    fig_override = fig_override)
   







############### WARNING: The functions below this line are more provisional.
# I either haven't used them in a while or created them quickly and did not
# yet get to fully testing, documenting, and polishing them.



#### Functions for representing / obtaining info about a particular article
def get_article_summary(df, ind):
    out_dict = {}
    out_dict["title"] = df["dc:title"].iloc[ind]
    out_dict["abstract"] = df["dc:description"].iloc[ind]
    out_dict["year"] = df["Year"].iloc[ind]
    return(out_dict)

def get_subset_summary(df, ind_list):
    """ind_list in order of best to worst"""
    out_dict = {}
    for i, ind in enumerate(ind_list):  
        out_dict[i] = get_article_summary(df,ind)
    return(out_dict)


def get_topic_top_doc_dict(theta_mat, df, topd = 5):
    K = theta_mat.shape[0]
    #get order of theta's for each topic from best to worst
    order_mat = np.flip(np.argsort(theta_mat, axis = 1), axis =1) 

    #get summary of each top doc
    top_doc_dict = {}
    for k in range(K):
        top_ind = order_mat[k][:topd]
        top_vals = theta_mat[k][top_ind]
        top_doc_dict[k] = get_subset_summary(df, top_ind)
        top_doc_dict[k]["theta_vals"] = top_vals
        
    return(top_doc_dict)








# Functions for looking at dataset with multiple journals and getting journal
#level information

def journal_breakdown(df, thresh = 1000):
    """
    
    Parameters
    ----------
    df : pandas dataframe
        must contain "Journal" column
    
    thresh :  int 
    
    Returns
    -------
    tuple with 
        0. number of journals
        1. the number of journals with nan value
        2. dict with journal names as keys and number of observations per
            journal as values
        3. dict with only journals with over thresh observations
      
    """
    
    
    journals = df.loc[:,"prism:publicationName"].copy()
    journals_dropna = journals.dropna()

    #must remove nan first for uniques function to work
    unique_journals = np.unique(journals_dropna)
    num_journals = len(unique_journals)
    
    num_nan = len(journals) - len(journals_dropna)
    
    #create dictionary with journal names as keys
    journal_dict = {}
    for elem in unique_journals:
        journal_dict[elem] = 0

    #loop to fill dictionary
    for j in range(0, len(journals_dropna)):
        if journals_dropna.iloc[j] == "nan": #shouldn't be any but just 
            print("Warning: there are still nan's in journals_dropna!")
            pass
        else:
            journal_dict[journals_dropna.iloc[j]] += 1
        
    #gather most frequent journals
    top_dict = {}
    for elem in journal_dict:
        if journal_dict[elem] > thresh:
            top_dict[elem] = journal_dict[elem]
        
    return(num_journals, num_nan, journal_dict, top_dict)
    
    
    
def journal_range(df, journal_dict, journal):
    """
    helper function to extract year range for a
    given journal
    
    Parameters
    ----------
    df : pandas dataframe
        full dataframe containing a Year and a prism:publicationName column
    journal_dict : dictionary
        journal_dict as output by journal_breakdown()
    title : string
        journal title

    Returns
    -------
    tuple containing earliest and latest year observed for given
    journal

    """
    subset = df[df.loc[:,"prism:publicationName"] == journal]
    subset_years = subset.Year.dropna()
    min_year = subset_years.min()
    max_year = subset_years.max()
    null_abstracts = np.sum(subset.loc[:,"dc:description"].isnull())
    return(min_year, max_year, null_abstracts)




def journal_summary(df, journal_dict, verbose = False):
    """
    Parameters
    ----------
    df : data frame
        must contain Year and prism:publicationName column
        
    journal_dict : dictionary
        journal dictionary as output by journal_breakdown()

    Returns
    -------
    pandas data frame containing columns for journal title, frequency,
    minimum year, maximum year, and year range

    """
    journal_titles = list(journal_dict.keys())
    nt = len(journal_dict)
    
    freq_list = [None] * nt
    min_list =  [None] * nt
    max_list =  [None] * nt
    null_abs_count = [None] * nt 

    for i, elem in enumerate(journal_titles):
        if (verbose and i % 10 == 0): #progress tracker
            print(i)
        freq_list[i] = journal_dict[elem]
        min_list[i], max_list[i], null_abs_count[i] = journal_range(df,
                                                                    journal_dict,
                                                                    elem)

    freq_journals = pd.DataFrame({"Journal":journal_titles, 
                             "Frequency":freq_list,
                             "MissingAbs": null_abs_count,
                             "MinYear":min_list,
                             "MaxYear":max_list,
                             "YearRange": np.array(max_list) - np.array(min_list) + 1
                             })
    
    return(freq_journals)



def journal_abstract_breakdown(df, journal, plot = True):
    """
    
    Parameters
    ----------
    df : pandas data frame
        must contain a "Journal" column
        
    journal : string
        name of journal in Journal column of df

    Returns
    -------
    subset - observations from df for given journal
    year_dict and year_percents for missingness by year
            as output by abstract_breakdown() function 


    Dependencies
    ------------
    requires abstract_breakdown() 
    """
    subset = df[df.loc[:,"prism:publicationName"] == journal]
    
    year_dict, year_percents = abstract_breakdown(subset, plot = plot, title = journal)
    
    
    
    return(subset, year_dict,year_percents)








