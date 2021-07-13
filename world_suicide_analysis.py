# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 19:46:43 2021

@authors: Alice Bogdan (ucn3qn), Kin Girma (zcu4kb), Anh Nguyen (hzn3kf), 
Frank Vasquez (fav3ba)

"""
import pandas as pd


# read in spreadsheets:
path = "World Happiness Report 2015-2016/"
df2015 = pd.read_csv(path+"2015.csv")
df2016 = pd.read_csv(path+"2016.csv")
dfsuicide = pd.read_csv("who_suicide_statistics.csv")

# add year column to happiness data
df2015['year'] = 2015
df2016['year'] = 2016

# combine happiness data
happy = df2015.append(df2016)

# drop statistical columns (some nans)
happy = happy.drop(['Standard Error','Lower Confidence Interval','Upper Confidence Interval'],axis=1)

# get 2015-2016 from suicide data
suic_filt = dfsuicide[(dfsuicide['year'] == 2015) | (dfsuicide['year'] == 2016)]

# drop nan values
suic_filt = suic_filt.dropna()

# drop sex and age
suic_filt = suic_filt.drop(['sex','age'], axis=1)

# renaming for agreement across dataframes
suic_filt = suic_filt.rename(columns={'country':'Country'})
suic_filt = suic_filt.replace({'United States of America':'United States'})

# sum population and suicides per country per year
suic_red = suic_filt.groupby(['Country', 'year'])['suicides_no','population'].agg('sum')

# add per capita column
suic_red['per_capita'] = suic_red.suicides_no/suic_red.population
  
# merge data into one frame
whole_df = pd.merge(suic_red,happy,on=['Country','year'])  

"""
TO DO:
    
process/combine sheets (Finished):
1. Eliminate irrelevant years
2. Join dataframes on country and year
3. Merge age buckets or keep separate?
    a. Merging would allow for simpler country to country comparison
    b. Leaving separate would give us more granular data, could see trends that
       might otherwise be masked.

Analysis:
Need to decide what exactly we want to do here
Start with playing with data and seeing if there are any obvious trends?
Maybe some visualizations would be useful as a start
Response variable (y): suicides
Predictor variables (xi): World Happiness Survey categories
 

Visualizations:
might end up being integrated throughout the rest of our code base
could make functions for visualizations that are called elsewhere?
"""



