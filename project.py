#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 19:12:43 2021

\@Healthy Happy Hoos
"""
# read in spreadsheets:
#suic_df = pd.read_csv()
#hap_df = pd.read_csv()

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

import pandas as pd

#import datasets for happiness and suicide data
df2015 = pd.read_csv("2015.csv")
df2016 = pd.read_csv("2016.csv")
dfsuicide = pd.read_csv("who_suicide_statistics.csv")

#filter for years 2015 and 2016
df2s = dfsuicide[(dfsuicide['year'] == 2015) | (dfsuicide['year'] == 2016)]

#drop columns sex and age
df2s = df2s.drop(['sex','age'], axis=1)

#drop na values
df2s = df2s.dropna()



#create an additional column in suicide data for suicides per capita
#dfsuicide['suicides_per_capita'] = dfsuicide[]


#merge data
#df1.merge(df2, left_on='lkey', right_on='rkey')
