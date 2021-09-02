# -*- coding: utf-8 -*-
"""

This file loads and explores the Demography data. It also removes
entries with missing or non-english abstracts and saves the final dataset

Because some processes (especially language detection) take some time to run,
it is written not to re-generate data and language detection file each time it is 
run. Instead, it tries to load the necessary files and only if not available
does it generate them and save them.

Author: Kyla Chasalow
#import os
Last edited: July 23, 2021
"""
#standard imports
import sys
import os
import pandas as pd
import numpy as np


# Import relative paths
from filepaths import code_path
from filepaths import demog_data_path as data_path
from filepaths import demog_eda_plots  as plot_path

#import other modules
sys.path.insert(0, code_path)
import EdaHelper
from Helpers import file_override_request


#Override setting
# ask user to set whether want script to override existing files of same name or not
override = file_override_request()

#for saving figures
dpi = 200

#EXTRACTING YEAR INFORMATION IF HAVE NOT ALREADY

try:
    TM_data = pd.read_csv(os.path.join(data_path,'Demog_data_original_withyears.csv'))
    print("Dataset with years shape:", TM_data.shape)
    
except:
    #Test if dataset required for analysis is available
    try:
        #Not including below: , index_col = 0) 
        #Don't use original indexing - leave it as a column so that can link up to bigger dataset if needed
        original_data = pd.read_csv(os.path.join(data_path, "Demog_data_original.csv"))
        print("Original data shape: ", original_data.shape)
    except:
        message =  "Demog_data_original_withyears.csv does not exist \n and Demog_data_original.csv"
        message = message + " does not exist to extract years from"
        raise Exception(message)

    print("Generating and saving dataset with years")
    #load original dataset  of Demography observations
    
    #obtain years and add to dataframe
    years = EdaHelper.extract_prism_year(original_data)
    TM_data = original_data.copy()
    TM_data = pd.concat([TM_data, pd.DataFrame(years)], axis = 1)
    TM_data.rename(columns = {0:'Year'}, inplace = True)

    print("Dataset with years shape:", TM_data.shape)
    TM_data.to_csv(os.path.join(data_path, "Demog_data_original_withyears.csv"), index = False)     






# MISSINGINGESS

#Missingness summaries for all variables
summary = EdaHelper.missing_summary(TM_data, 
                                    save = True,
                                    override = override, 
                                    filename = "Demog_missing_summary",
                                     outpath = data_path) 
print("------------------------------")
print("Missingness Summary")
print(summary)
print("------------------------------")


#Missingness of abstracts by type
abs_sum, word_count = EdaHelper.missing_abstract_summary(TM_data, save = True,
                                             override = override,
                                             filename = "Demog_missing_abstract_summary",
                                             outpath = data_path
                                             )
print("------------------------------")
print("Summary of Missing Abstracts by Type")
print(abs_sum)
print("Of the  %d missing abstracts for entries of type 'article'" % abs_sum.loc["Article","num_missing"],
      " %d contain the words \n erratum, comment, response, addendum, reply, or correction" % word_count)



# COUNTS BY YEAR (before removing any) - generate plot


out_dict = EdaHelper.year_breakdown(TM_data.Year,
                                        plot = True,
                                        figsize = (15,4),
                                        title = "Number of Observations per Year: Demography",
                                        save_fig = True,
                                        fig_outpath = plot_path,
                                        dpi = dpi,
                                        fig_name = "Observations_by_year_preprune",
                                        fig_override = override)



print("------------------------------")
print("missing years:", out_dict["num_missing"],
      "\n minimum year: ", out_dict["min_year"],
      "\n maximum year: ", out_dict["max_year"], 
      "\n year gaps :", out_dict["year_gaps"],
      "\n earliest continuous year: ", out_dict["earliest_continuous_year"])
    
         

# PERCENT OF PAPERS MISSING ABSTRACTS BY YEAR

year_dict, year_percents = EdaHelper.abstract_breakdown(TM_data, plot = True,
                                                       title = "Percent of Observations Missing Abstracts: Demography",
                                                       save_fig = True,
                                                       fig_name = "Demog_abstract_missingness",
                                                       fig_override = override,
                                                       dpi = dpi,
                                                       fig_outpath = plot_path)

print("------------------------------")
print("There are ", year_dict['nan'], "observations with missing abstract and missing year")
n_missing = TM_data.loc[:,"dc:description"].isnull().sum()
n = TM_data.shape[0]
print("There are ", n_missing, " or ", np.round(100*n_missing/n, 3), "% observations missing abstracts")

miss = np.sum(np.array([year_dict[c] for c in range(1964,1970)]))
print( miss," of the missing abstracts come from 1964-1969")
miss = np.sum(np.array([year_dict[c] for c in range(1964,1980)]))
print( miss," of the missing abstracts come from 1964-1979")




# LANGUAGE DETECTION
print("------------------------------")
try:
    print("loading languages file")
    abs_lang = np.load(os.path.join(data_path,"Demog_languages.npy"), allow_pickle = True)

except:
    print("language file not available - generating language file using langid language detection")
    print("Warning: this may take a while...")
    abs_lang = EdaHelper.abstract_languages(TM_data, thresh = 1000, verbose = True)
    np.save(os.path.join(data_path,"Demog_languages.npy"), np.array(abs_lang))

num_en, num_not_en, num_na, en_list, not_en_list, na_list, error_count = EdaHelper.lang_breakdown(abs_lang)
print(num_en, " labeled English: ", round(100*num_en/n,3), "%")
print(num_na, " are NA: ", round(100*num_na/n,3), "%")
print(num_not_en, " labeled Not English: ", round(100*num_not_en/n,3), "%")
print(error_count, " langid errors")


ind = np.where(TM_data.keys() == "dc:description")[0][0]
print("Non-english abstracts are: \n")
print("language:", abs_lang[not_en_list])
print("entries:", TM_data.iloc[not_en_list,[ind, 35]])



# FINAL DATASET
print("----------------------------------")

try:
    TM_data_final = pd.read_csv(os.path.join(data_path,"Demog_data_final.csv"))
    print("dropping %d non-English entries" % len(not_en_list))
    print("dropping missing abstracts")
except:
    print("dropping %d non-English entries" % len(not_en_list))
    TM_data_en = TM_data.drop(not_en_list)
    print("dropping missing abstracts")
    TM_data_final = TM_data_en.dropna(subset=['dc:description'])
    print("saving final dataset")
    TM_data_final.to_csv(os.path.join(data_path,"Demog_data_final.csv"), index = False)

print("final dataset has shape:", TM_data_final.shape)


print("----------------------------------")


# COUNTS BY YEAR (after pruning) - generate plot

num_missing, min_year, max_year, continuous, year_counts = EdaHelper.year_breakdown(TM_data_final.Year, 
                                                                          plot = True,
                                                                          title = "Number of Abstracts per Year: Demography",
                                                                          figsize = (15,4),
                                                                          min_plot_year = 1964, #match the earlier graph
                                                                          save_fig = True,
                                                                          fig_outpath = plot_path,
                                                                          dpi = dpi,
                                                                          fig_name = "Observations_by_year_postprune",
                                                                          fig_override = override)

print("------------------------------")
print( "After pruning:"
      "\n minimum year: ", min_year,
      "\n maximum year: ", max_year, 
      "\n continuous? ", continuous)


#Abstract lengths in characters using final data
summary_dict = EdaHelper.abstract_length_breakdown(TM_data_final, 
                                                   plot = True,
                                                   title = "Abstract Lengths by Year: Demography",
                                                   color = "orange",
                                                   save_fig = True,
                                                   fig_outpath = plot_path,
                                                   dpi = dpi,
                                                   fig_name = "Abstract_Length_breakdown",
                                                   fig_override = override)




######################################################################################