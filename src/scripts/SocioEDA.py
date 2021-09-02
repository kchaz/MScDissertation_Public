# -*- coding: utf-8 -*-
"""

This file loads and explores the Sociology data. It also removes
entries with missing or non-english abstracts and entries from pre-1996 
and saves a final dataset

Because some processes (especially language detection) can take some time to run,
(though more for larger datasets) it is written not to re-generate data and 
detection file each time it is run. Instead, it tries to load the necessary files 
and only if not available does it generate them and save them.

Author: Kyla Chasalow
Last edited: August 12, 2021
"""
#standard imports
import sys
import os
import pandas as pd
import numpy as np


# Import relative paths
from filepaths import code_path
from filepaths import socio_data_path as data_path
from filepaths import socio_eda_plots  as plot_path


#import other modules
sys.path.insert(0, code_path)
import EdaHelper
from Helpers import file_override_request


#Override setting
# ask user to set whether want script to override existing files of same name or not
override = file_override_request()

#for saving figures
dpi = 200




#------------------------------------------------------------------------------

#EXTRACT YEAR INFORMATION FOR ALL OBSERVATIONS

try:
    TM_data = pd.read_csv(os.path.join(data_path,'Socio_data_original_withyears.csv'))
    print("Dataset with years shape:", TM_data.shape)
    
except:
    #Test if dataset required for analysis is available
    try:
        #Not including below: , index_col = 0) 
        #Don't use original indexing - leave it as a column so that can link up to bigger dataset if needed
        original_data = pd.read_csv(os.path.join(data_path, "Socio_data_original.csv")) 
        print("Original data shape: ", original_data.shape)
    except:
        message =  "Socio_data_original_withyears.csv does not exist \n and Socio_data_original.csv"
        message = message + " does not exist to extract years from"
        raise Exception(message)

    print("Generating and saving dataset with years")
    #load original dataset  of Socioraphy observations
    
    #obtain years and add to dataframe
    years = EdaHelper.extract_prism_year(original_data)
    TM_data = pd.concat([original_data, pd.DataFrame(years)], axis = 1)
    TM_data.rename(columns = {0:'Year'}, inplace = True)

    print("Dataset with years shape:", TM_data.shape)
    TM_data.to_csv(os.path.join(data_path, "Socio_data_original_withyears.csv"), index = False)     





#------------------------------------------------------------------------------

# DEFINE A DATASET FOR EACH JOURNAL

ASR = TM_data[ TM_data["prism:publicationName"] == "American Sociological Review"]
AJS = TM_data[ TM_data["prism:publicationName"] == "American Journal of Sociology"]
ARS = TM_data[ TM_data["prism:publicationName"] == "Annual Review of Sociology"] 
names = ["American Sociological Review",
         "American Journal of Sociology", 
         "Annual Review of Sociology"]
abbrev = ["ASR", "AJS", "ARS"]
journal_data_list = [ASR, AJS, ARS]

print("Number of observations by journal - before removing anything")
print(names[0], ": ", ASR.shape)
print(names[1], ": ",  AJS.shape)
print(names[2], ": ",  ARS.shape)
print("\n\n\n\n\n")


#------------------------------------------------------------------------------
####### BEFORE CUTTING DOWN TO POST-1996
#------------------------------------------------------------------------------
print("BEFORE MAKING ANY CUTS")
print("----------------------\n----------------------\n----------------------\n\n")

# MISSINGINGESS (BY JOURNAL AND OVERALL) 

#Missingness summaries for all variables
summary = EdaHelper.missing_summary(TM_data, 
                                     save = True,
                                     override = override, 
                                     filename = "Socio_missing_summary",
                                     outpath = data_path) 
print("Missingness Summary of All Journals")
print(summary)

print("------------------------------")
print("Missing Astracts per Journal")
for i, data in enumerate(journal_data_list):
    print(names[i], ": ", EdaHelper.missing_summary(data, save = False ).iloc[26,:])
    
    

#Missingness of abstracts by type
print("------------------------------")
print("Missinginess of Abstracts by Type")

for i, data in enumerate(journal_data_list):    
    abs_sum, word_count = EdaHelper.missing_abstract_summary(data, save = False)
    print("\n"  + names[i])
    print(abs_sum)
    print("Of the  %d missing abstracts for entries of type 'article'" % abs_sum.loc["Article","num_missing"],
       " %d contain the words \n erratum, comment, response, addendum, reply, or correction" % word_count)



# COUNTS BY YEAR 
print("------------------------------")
print("Generating Plots of Observation Counts by Year - BEFORE removing missing abstracts \n ")
print("------------------------------")

for i, data in enumerate(journal_data_list):
    out_dict = EdaHelper.year_breakdown(data.Year,
                                        plot = True,
                                        figsize = (15,4),
                                        title = "Number of Observations by Year: %s" % names[i],
                                        save_fig = False)
    
    print("\n" , names[i])
    print("missing years:", out_dict["num_missing"],
          "\n minimum year: ", out_dict["min_year"],
          "\n maximum year: ", out_dict["max_year"], 
          "\n year gaps :", out_dict["year_gaps"],
          "\n earliest continuous year: ", out_dict["earliest_continuous_year"])
    
#generate the plots in grids
EdaHelper.by_year_grid(TM_data, plottype = "counts",
                              suptitle = "Number of Observations per Year",
                              save_fig = True,
                              fig_outpath = plot_path,
                              dpi = dpi,
                              fig_name = "socio_obs_by_year_pre_prune",
                              fig_override = override)
    
    

print("------------------------------")
print("Generating plots for missing abstracts by year")
#generate the plots in grids
EdaHelper.by_year_grid(TM_data, plottype = "missing",
                              suptitle = None, #use default
                              save_fig = True,
                              fig_outpath = plot_path,
                              dpi = dpi,
                              fig_name = "socio_missing_by_year_pre_prune",
                              fig_override = override)
    

# LANGUAGE
print("\n\n\n------------------------------")
print("Language Detection")

try:
    print("loading languages file")
    abs_lang = np.load(os.path.join(data_path,"Socio_languages.npy"), allow_pickle = True)

except:
    print("language file not available - generating language file using langid language detection")
    abs_lang = EdaHelper.abstract_languages(TM_data, thresh = 1000, verbose = True)
    np.save(os.path.join(data_path,"Socio_languages.npy"), np.array(abs_lang))


n = TM_data.shape[0]
num_en, num_not_en, num_na, en_list, not_en_list, na_list, error_count = EdaHelper.lang_breakdown(abs_lang)
print(num_en, " labeled English: ", round(100*num_en/n,3), "%")
print(num_na, " are NA: ", round(100*num_na/n,3), "%")
print(num_not_en, " labeled Not English: ", round(100*num_not_en/n,3), "%")
print(error_count, " langid errors")






#TO DROP:
# * anything not in English
# * missing abstracts
# * any observatiosn pre-1996 because I have outside evidence that the SCOPUS
#   data here is very incomplete (for ARS, there are no observations there!)

print("----------------------------------")

#do all this even though not saving it if TM_data_final already saved because
#outputs useful information about process of getting to final data


#handle non-English if needed
if num_not_en > 0:
      print("dropping %d non-English entries" % len(not_en_list))
      TM_data = TM_data.drop(not_en_list)

n1 = TM_data.shape[0]    



#drop pre-1996    
print("dropping pre-1996 observations")
TM_data_final = TM_data[TM_data.Year >= 1996]
n2 = TM_data_final.shape[0]
print("%d observations dropped" % (n1-n2))

print("Missingness for remaining observations:")
abs_sum, word_count = EdaHelper.missing_abstract_summary(TM_data_final, save = False)
print(abs_sum)
print("Of the  %d missing abstracts for entries of type 'article'" % abs_sum.loc["Article","num_missing"],
   " %d contain the words \n erratum, comment, response, addendum, reply, or correction" % word_count)
print("\n\n\n\n")



#drop missing abstracts
print("dropping missing abstracts")
TM_data_final = TM_data_final.dropna(subset=['dc:description'])
n3 = TM_data_final.shape[0]
print("%d observations dropped" % (n2-n3))



try:
    TM_data_final = pd.read_csv(os.path.join(data_path,"Socio_data_final.csv"))
    
    
except:
    print("\n")
    print("saving final dataset")
    TM_data_final.to_csv(os.path.join(data_path,"Socio_data_final.csv"), index = False)

    

print("final dataset has shape:", TM_data_final.shape)





#------------------------------------------------------------------------------
####### AFTER CUTTING DOWN TO POST-1996
#------------------------------------------------------------------------------
print("FINAL DATASET ANALYSIS")
print("----------------------\n----------------------\n----------------------\n\n")


#BREAKING UP FINAL DATASET BY JOURNAL

# DEFINE A DATASET FOR EACH JOURNAL

ASR_final = TM_data_final[ TM_data_final["prism:publicationName"] == "American Sociological Review"]
AJS_final = TM_data_final[ TM_data_final["prism:publicationName"] == "American Journal of Sociology"]
ARS_final = TM_data_final[ TM_data_final["prism:publicationName"] == "Annual Review of Sociology"] 
journal_data_list_final = [ASR_final, AJS_final, ARS_final]

print("Number of observations by journal - final")
print(names[0], ": ", ASR_final.shape)
print(names[1], ": ",  AJS_final.shape)
print(names[2], ": ",  ARS_final.shape)
print("\n\n\n\n\n")



print("----------------------------------")
print("Examining abstract counts by year after removal")

for i, data in enumerate(journal_data_list_final):
    out_dict = EdaHelper.year_breakdown(data.Year,
                                        plot = True,
                                        figsize = (15,4),
                                        title = "Number of Abstracts by Year: %s" % names[i],
                                        save_fig = False)
    print(names[i])
    print("missing years:", out_dict["num_missing"],
          "\n minimum year: ", out_dict["min_year"],
          "\n maximum year: ", out_dict["max_year"], 
          "\n year gaps :", out_dict["year_gaps"],
          "\n earliest continuous year: ", out_dict["earliest_continuous_year"])

#generate figures in a grid
EdaHelper.by_year_grid(TM_data_final, plottype = "counts",
                              suptitle = "Number of Abstracts per Year", #use default
                              save_fig = True,
                              fig_outpath = plot_path,
                              dpi = dpi,
                              fig_name = "socio_obs_by_year_post_prune0",
                              fig_override = override)



