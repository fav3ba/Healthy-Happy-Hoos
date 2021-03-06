# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 19:46:43 2021

@authors: Alice Bogdan (ucn3qn), Kin Girma (zcu4kb), Anh Nguyen (hzn3kf), 
Frank Vasquez (fav3ba)

"""
import pandas as pd
import DecisionTree

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

#filter for countries with both happiness and suicide data in 2015 and 2016
whole_df_2 = whole_df.groupby("Country").filter(lambda x: x.Country.size == 2)

whole_df['per_100k'] = whole_df['per_capita']*100000

regions = list(whole_df['Region'].unique())

# make region values numerical for training
region_factor = []
for i in range(0, len(whole_df)):
    reg = whole_df.iloc[i,5]
    factor = [i for i in range(len(regions)) if regions[i] == reg]
    region_factor.append(factor[0])

whole_df['region_factor'] = region_factor

#create buckets (0-10, 10-20, 20-30, >30)
buckets = []
for i in range(0, len(whole_df)):
    val = whole_df.iloc[i,15]
    if(val < 10): buckets.append('0-10')
    elif (val < 20 and val >= 10): buckets.append('10-20')
    elif(val < 30 and val >= 20): buckets.append('20-30')
    elif (val >= 30): buckets.append('30+')
    
whole_df['buckets'] = buckets

# input data and desired tree depth
model = DecisionTree.build_model(whole_df, 10)



