# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 19:46:43 2021

@authors: Alice Bogdan (ucn3qn), Kin Girma (zcu4kb), Anh Nguyen (hzn3kf), 
Frank Vasquez (fav3ba)

"""
import pandas as pd


# read in spreadsheets:
suic_df = pd.read_csv()
hap_df = pd.read_csv()

#checking for branch functionality

"""
process/combine sheets:
1. Eliminate irrelevant years
2. Join dataframes on country and year
3. Merge age buckets or keep separate?
    a. Merging would allow for simpler country to country comparison
    b. Leaving separate would give us more granular data, could see trends that
       might otherwise be masked.
"""

"""
Analysis:
Need to decide what exactly we want to do here
Start with playing with data and seeing if there are any obvious trends?
Maybe some visualizations would be useful as a start
Response variable (y): suicides
Predictor variables (xi): World Happiness Survey categories
 
"""

"""
Visualizations:
might end up being integrated throughout the rest of our code base
could make functions for visualizations that are called elsewhere?
"""



