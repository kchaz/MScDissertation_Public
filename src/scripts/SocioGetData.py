# -*- coding: utf-8 -*-
"""

WARNING: THIS FILE DOES NOT USE RELATIVE FILEPATHS BECAUSE
I LOAD THE DATA FROM AN EXTERNAL DRIVE 
    
author: Kyla Chasalow

"""
import pandas as pd
import os
from filepaths import socio_data_path

soci_data = pd.read_csv('F:\DissertationData\scopus_search_SOCI_20210518.tsv', sep='\t')

journals_list = [
           "American Sociological Review",
           "American Journal of Sociology",
           "Annual Review of Sociology"
           ]

subset  = soci_data[soci_data["prism:publicationName"].isin(journals_list)]
subset = subset.reset_index() #will add index as a column

subset.to_csv(os.path.join(socio_data_path, "Socio_data_original.csv"), index = False)     