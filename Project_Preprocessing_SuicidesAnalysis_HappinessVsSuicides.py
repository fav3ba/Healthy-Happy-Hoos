#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alicebogdan
"""

'''
For reference:
    world: contains country/continent info important for mapping
    dfregion: contains 2 columns with countries and regions extracted from df2015
    suic_red: suicide data without sex and age columns. No region, country, and continent (see suicides)
    suicides: contains suicide info with region, country, and continent without nan values
    happy: contains happiness data for years 2015 and 2016 (contains region, country, and continent info)
    happy_all: same as happy but also contains years 2017-2019
    happy_sad: combination of happy_all and suic_red
    whole_world: copy of happy_sad
    part_world: contains only countries that had data for both 2015 and 2016
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import matplotlib.pylab as plt
import plotly.graph_objects as go
import plotly.express as px
import math
import geopandas as gpd
from pandas.api.types import CategoricalDtype
from statistics import stdev

# read in spreadsheets:
path = "World Happiness Report 2015-2016/"
df2015 = pd.read_csv(path+"2015.csv")
df2016 = pd.read_csv(path+"2016.csv")
df2017 = pd.read_csv(path+"2017.csv")
df2018 = pd.read_csv(path+"2018.csv")
df2019 = pd.read_csv(path+"2019.csv")
dfsuicide = pd.read_csv("who_suicide_statistics.csv")
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

##############################################################################
#                   DATA PREPROCESSING
##############################################################################

######################### World #####################################################
#clean world dataframe (for mapping visualization)
#rename columns for agreement across dataframes
world = world.rename(columns = {'continent':'Continent'})
world = world.rename(columns = {'name':'Country'}) 
world = world.rename(columns = {'iso_a3':'Country_Code'})
len(world) #177

#drop Antarctica
world = world[world.Country!="Antarctica"]
len(world) #176

#rename Czechia to Czech Republic
world['Country'] = world['Country'].replace({'Czechia': 'Czech Republic'})

#remove population and gdp columns
world2 = world.copy()
world = world.drop(['pop_est','gdp_md_est'], axis = 1)

######################## Region ######################################################
#create region df
dfregion = df2015[['Country','Region']]
dfregion = dfregion.replace({'Central African Republic':'Central African Rep.',
                               'Congo (Brazzaville)':'Congo',
                               'Bosnia and Herzegovina':'Bosnia and Herz.',
                               'Congo (Kinshasa)':'Dem. Rep. Congo',
                               'Dominican Republic':'Dominican Rep.',
                               'North Cyprus':'N. Cyprus',
                               'Palestinian Territories':'Palestine',
                               'Sudan':'S. Sudan',
                               'Somaliland region':'Somaliland',
                               'United States':'United States of America'})

############################ Happiness ##################################################
#clean happiness data
# add year column to happiness data
df2015['year'] = 2015
#print(len(df2015))
df2016['year'] = 2016
#print(len(df2016))
df2017['year'] = 2017
#print(len(df2017))
df2018['year'] = 2018
#print(len(df2018))
df2019['year'] = 2019
#print(len(df2019))

#remove inconsistent columns
df2015 = df2015.drop(['Standard Error', "Dystopia Residual","Region"], axis=1)
df2016 = df2016.drop(['Lower Confidence Interval','Upper Confidence Interval',"Dystopia Residual","Region"], axis=1)
df2017 = df2017.drop(['Whisker.high','Whisker.low','Dystopia.Residual'], axis=1)

#rename column names for consistency
df2017 = df2017.rename(columns = {"Economy..GDP.per.Capita.":"Economy (GDP per Capita)",
                                  "Happiness.Rank":"Happiness Rank",
                                  "Happiness.Score":"Happiness Score",
                                  "Health..Life.Expectancy.":"Health (Life Expectancy)",
                                  "Trust..Government.Corruption.":"Trust (Government Corruption)"})
df2018 = df2018.rename(columns = {'Country or region':"Country",
                                  "Freedom to make life choices":"Freedom",
                                  "GDP per capita":"Economy (GDP per Capita)",
                                  "Score":"Happiness Score",
                                  "Healthy life expectancy":"Health (Life Expectancy)",
                                  "Perceptions of corruption":"Trust (Government Corruption)",
                                  "Overall rank":"Happiness Rank",
                                  "Social support":"Family"})
df2019 = df2019.rename(columns = {'Country or region':"Country",
                                  "Freedom to make life choices":"Freedom",
                                  "GDP per capita":"Economy (GDP per Capita)",
                                  "Score":"Happiness Score",
                                  "Healthy life expectancy":"Health (Life Expectancy)",
                                  "Perceptions of corruption":"Trust (Government Corruption)",
                                  "Overall rank":"Happiness Rank",
                                  "Social support":"Family"})

# combine happiness data (2015-2016)
happy = df2015.append(df2016)
#print(len(happy))
#sorted(happy)

#rename countries
happy = happy.replace({'Central African Republic':'Central African Rep.',
                               'Congo (Brazzaville)':'Congo',
                               'Bosnia and Herzegovina':'Bosnia and Herz.',
                               'Congo (Kinshasa)':'Dem. Rep. Congo',
                               'Dominican Republic':'Dominican Rep.',
                               'North Cyprus':'N. Cyprus',
                               'Palestinian Territories':'Palestine',
                               'Sudan':'S. Sudan',
                               'Somaliland region':'Somaliland',
                               'Somaliland Region':'Somaliland',
                               'United States':'United States of America'})

#combine all happiness data (2015-2019)
happy_all = happy.append(df2017)
happy_all = happy_all.append(df2018)
happy_all = happy_all.append(df2019)

#rename countries
happy_all = happy_all.replace({'Central African Republic':'Central African Rep.',
                               'Congo (Brazzaville)':'Congo',
                               'Bosnia and Herzegovina':'Bosnia and Herz.',
                               'Congo (Kinshasa)':'Dem. Rep. Congo',
                               'Dominican Republic':'Dominican Rep.',
                               'North Cyprus':'N. Cyprus',
                               'Palestinian Territories':'Palestine',
                               'Sudan':'S. Sudan',
                               'Somaliland region':'Somaliland',
                               'Somaliland Region':'Somaliland',
                               'United States':'United States of America'})

#add region and continents to happy dfs
happy = pd.merge(happy, dfregion, how = 'left', on = 'Country')
happy = pd.merge(happy, world, how ='left', on = 'Country')

happy_all = pd.merge(happy_all, dfregion, how = 'left', on = 'Country')
happy_all = pd.merge(happy_all, world, how ='left', on = 'Country')
happy_all_dropna = happy_all.dropna()
len(sorted(happy_all['Country'].unique()))

#average happiness score for each country
happy_all_average = happy_all.groupby(['Country'])['Happiness Score'].mean()
#reindex
happy_all_average = happy_all_average.reset_index()
#add regions
happy_all_average = pd.merge(happy_all_average, dfregion, how = 'left', on = 'Country')
#add mapping data
happy_all_average = pd.merge(happy_all_average, world, how ='left', on = 'Country')
#drop na values
happy_all_average = happy_all_average.dropna()

#sorted(dfregion['Region'].unique())

#rename for agreement across dataframes (world df used United States of America)
#happy_all = happy_all.replace({'United States':'United States of America'})

############################# Suicides #################################################
#clean suicide data
#rename countries to match world dataframe
dfsuicide = dfsuicide.replace({'Russian Federation':'Russia',
                               'Serbia ':'Serbia',
                               'Bosnia and Herzegovina':'Bosnia and Herz.',
                               'Iran (Islamic Rep of)':'Iran',
                               'Republic of Korea':'South Korea',
                               'Republic of Moldova':'Moldova',
                               'TFYR Macedonia':'Macedonia',
                               'Venezuela (Bolivarian Republic of)':'Venezuela',
                               'Brunei Darussalam':'Brunei'})

#drop nan values
suic_fil = dfsuicide.dropna()

#renaming for agreement across dataframes
suic_fil = suic_fil.rename(columns = {'country':"Country"})
suic = suic_fil.copy()

# drop sex and age
suic_fil = suic_fil.drop(['sex','age'], axis=1)
# sum population and suicides per country per year
suic_red = suic_fil.groupby(['Country', 'year'])['suicides_no','population'].agg('sum')
#reset so Country and year aren't the index
suic_red = suic_red.reset_index()

# add per capita column
suic_red['per_capita'] = suic_red.suicides_no/suic_red.population
suic_red['Suicides per 100K'] = round(suic_red.per_capita*100000,3)

#add countries to suicides 
suicides = pd.merge(suic_red, dfregion, how = 'left', on = 'Country')
suicides = pd.merge(suicides, world, how = 'left', on = 'Country')

#drop nan values
suicides = suicides.dropna()

#add region and countries to suic
suic = pd.merge(suic, dfregion, how = 'left', on = 'Country')
suic = pd.merge(suic, world, how = 'left', on = 'Country')
suic = suic.dropna()

#clean suic
#remove years from age
suic.loc[:, 'age'] = suic['age'].str.replace(' years','')

##############################################################################
#set age as categorical variable
age_categories = ['5-14', '15-24', '25-34', '35-54', '55-74', '75+']
suic.age = suic.age.astype(CategoricalDtype(categories=age_categories, ordered=True))
suic_year_cont = suic.copy()
suic_gender = suic.copy()
suic_gender2 = suic.copy()
suic_average = suic.copy()
suic_age = suic.copy()

#add suicides per 100K
suic['per_capita'] = suic.suicides_no/suic_red.population
suic['Suicides per 100K'] = round(suic.per_capita*100000,3)

##############################################################################
#groupby year and continent
suic_year_cont = suic_year_cont.groupby(['year','Continent'])['suicides_no','population'].agg('sum')
#reset index
suic_year_cont = suic_year_cont.reset_index()
#calculate suicides per 100K
suic_year_cont['Suicides per 100K'] = round((suic_year_cont.suicides_no/suic_year_cont.population)*100000,3)

#mean suicide (11.09)
suic_year_cont['Suicides per 100K'].mean()

##############################################################################
#groupby year and age
suic_age = suic_age.groupby(['year','age'])['suicides_no','population'].agg('sum').reset_index()
#calculate suicides per 100K
suic_age['Suicides per 100K'] = round((suic_age.suicides_no/suic_age.population)*100000,3)

##############################################################################
#group suicides by gender and continent
suic_gender = suic_gender.groupby(['year','Continent','sex'])['suicides_no','population'].agg('sum')
#reset index
suic_gender = suic_gender.reset_index()
#calculate suicides per 100K
suic_gender['Suicides per 100K'] = round((suic_gender.suicides_no/suic_gender.population)*100000,3)

##############################################################################
#group suicides by gender and continent
suic_gender2 = suic_gender2.groupby(['year','sex'])['suicides_no','population'].agg('sum')
#reset index
suic_gender2 = suic_gender2.reset_index()
#calculate suicides per 100K
suic_gender2['Suicides per 100K'] = round((suic_gender2.suicides_no/suic_gender2.population)*100000,3)

##############################################################################
#average suicides per country
suic_average = suic_average.groupby(['year','Country'])['suicides_no','population'].agg('sum').reset_index()
suic_average['Country_count'] = 1
suic_average = suic_average.groupby(['Country']).agg({'suicides_no':'sum','population':'sum','Country_count':'count'}).reset_index()
suic_average['Suicides per 100K'] = round((suic_average.suicides_no/suic_average.population)*100000,3)
suic_average = pd.merge(suic_average, dfregion, how = 'left', on = 'Country')
suic_average = pd.merge(suic_average, world, how = 'left', on = 'Country')

#mean suicide (12.35)
suic_average['Suicides per 100K'].mean()

##############################################################################
#groupby year and country
suic_country_year = suic.groupby(['year','Country'])['suicides_no','population'].agg('sum').reset_index()
suic_country_year['Suicides per 100K'] = round((suic_country_year.suicides_no/suic_country_year.population)*100000,3)

#mean suicide for country across all years (13.22)
suic_mean = suic_country_year['Suicides per 100K'].mean()
#standard deviation
suic_std = stdev(suic_country_year['Suicides per 100K'])
#3 standard deviations
suic_std*3+suic_std

##############################################################################
#top 10 countries with avg suicde
top_10_suic_average = suic_average.sort_values(by = 'Suicides per 100K', ascending= False).head(10)

#lowest 10 countries with avg suicide
bottom_10_suic_average = suic_average.sort_values(by = 'Suicides per 100K', ascending= False).tail(10)

#countries with the most data
top_10_suic_count = suic_average.sort_values(by = 'Country_count', ascending= False).head(10)

#countries with the least data
low_10_suic_count = suic_average.sort_values(by = 'Country_count', ascending= False).tail(10)

############################### USSR ##############################################
#filter for USSR countries
USSR = ['Russia', 'Ukraine', 'Georgia', 'Belarus', 'Uzbekistan', 'Armenia', 'Azerbaijan', 'Kazakhstan', 'Kyrgyzstan', 'Moldova', 'Turkmenistan', 'Tajikistan', 'Latvia', 'Lithuania', 'Estonia']
USSR_suic =suic.copy()
#filter for USSR countries only
USSR_suic = USSR_suic[USSR_suic['Country'].isin(USSR)]
#groupby year and continent
USSR_suic = USSR_suic.groupby(['year','Country'])['suicides_no','population'].agg('sum').reset_index()
#calculate suicides per 100K
USSR_suic['Suicides per 100K'] = round((USSR_suic.suicides_no/USSR_suic.population)*100000,3)

##############################################################################
#filter for USSR as one
USSR_whole = suic.copy()
#filter for USSR countries only
USSR_whole = USSR_whole[USSR_whole['Country'].isin(USSR)]
#groupby year
USSR_whole = USSR_whole.groupby(['year'])['suicides_no','population'].agg('sum').reset_index()
#calculate suicides per 100K
USSR_whole['Suicides per 100K'] = round((USSR_whole.suicides_no/USSR_whole.population)*100000,3)

####################### Combined Happiness and Suicides################################
#merge suicide and happy dataframes together
happy_sad = pd.merge(suic_red, happy, on=['Country','year'])

#merge world and happy_sad dataframes together
whole_world = happy_sad.copy()
whole_world = whole_world.dropna()
len(whole_world.Country.unique())

#whole_world df filtered for countries that have entries for 2015 and 2016
part_world = whole_world.groupby('Country').filter(lambda x: x.Country.size == 2)
part_world = part_world.dropna()
sorted(part_world.Country.unique())

############################## Average Combined Data ###################################
#create df of average happiness and average suicides per 100k
#avg happiness for country
world_happy_avg = happy_all.groupby(['Country'])['Happiness Score'].mean().reset_index()

#avg suicides by country
world_suic_avg = suic_fil.copy()
world_suic_avg['Country_count'] = 1
world_suic_avg = world_suic_avg.groupby(['Country']).agg({'suicides_no':'sum','population':'sum','Country_count':'count'}).reset_index()
world_suic_avg['Suicides per 100K'] = round((world_suic_avg.suicides_no/world_suic_avg.population)*100000,3)
world_suic_avg['population'] = round((world_suic_avg.population/world_suic_avg.Country_count),0)

#whole_world_avg = pd.merge(world_suic_avg, world_happy_avg, on = ['Country'])
whole_world_avg = pd.merge(world_suic_avg, world_happy_avg, how = 'inner', on = 'Country')
whole_world_avg = pd.merge(whole_world_avg, dfregion, how = 'left', on = 'Country')
whole_world_avg = pd.merge(whole_world_avg, world2, how = 'left', on = 'Country')
whole_world_avg = whole_world_avg.dropna()
whole_world_avg.head()
#print(len(whole_world_avg)) #82


#outer joing for world and happy_sad dafaframes
#whole_world2 = pd.merge(happy_sad, world, how = "outer", on = 'Country')
#sorted(world['Country'].unique())
#sorted(happy_sad['Country'].unique())

##############################################################################
#                       VISUALIZATIONS
##############################################################################
#line plot of suicdes per 100K over time by continent (fairly constant numbers)
line_suic_cont = sns.lineplot(x='year', y='Suicides per 100K', hue ='Continent', data= suic_year_cont)
#move legend outside of graph
line_suic_cont.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#add title
line_suic_cont.set_title('Suicides per 100K Over Time, by Continent')
#change x label to be capitalized
line_suic_cont.set_xlabel('Year')

##############################################################################
#line plot of suicdes per 100K over time by gender (significantly more suicides for males than females)
line_suic_gender2 = sns.lineplot(x='year', y='Suicides per 100K', hue ='sex', data= suic_gender2)
#move legend outside of graph
line_suic_gender2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#add title
line_suic_gender2.set_title('Suicides per 100K Over Time, by Sex')
#change x label to be capitalized
line_suic_gender2.set_xlabel('Year')

#summary for gender
suic_gender2.loc[suic_gender['sex'] == 'female'].sum()
suic_gender2.loc[suic_gender['sex'] == 'male'].sum()

##############################################################################
#line plot of suicdes per 100K over time by age
line_suic_age = sns.lineplot(x='year', y='Suicides per 100K', hue ='age', data= suic_age)
#move legend outside of graph
#line_suic_age.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#add title
line_suic_age.set_title('Suicides per 100K Over Time, by Age')
#change x and y labels to be capitalized
line_suic_age.set_xlabel('Year')
line_suic_age.set_ylabel('Age')

##############################################################################
#barplot of top 10 countries with greatest suicides
barplot = sns.barplot(x = 'Country', y= 'Suicides per 100K', data = top_10_suic_average, palette = 'Blues_d').set_title('Countries with Highest Average Suicides per 100K')
plt.xticks(rotation=45)
plt.show

##############################################################################
#barplot with lowest countries of suicides
barplot2 = sns.barplot(x = 'Country', y= 'Suicides per 100K', data = bottom_10_suic_average, palette = 'Greens_d').set_title('Countries with Lowest Average Suicides per 100K')
plt.xticks(rotation=45)
plt.show

##############################################################################
#barplot of top 10 countries with highest data counts
suic_high = suic_average.sort_values(by = 'Country_count', ascending= False).head(20)
barplot_avgh = sns.barplot(x = 'Country', y= 'Country_count', data = suic_high, palette = 'Blues_d').set_title('Countries with Most Data')
plt.xticks(rotation=45)
plt.show

##############################################################################
#barplot of top 10 countries with highest data counts
suic_low = suic_average.sort_values(by = 'Country_count', ascending= False).tail(20)
barplot_avgl = sns.barplot(x = 'Country', y= 'Country_count', data = suic_low, palette = 'Reds_d').set_title('Countries with Most Data')
plt.xticks(rotation=45)
plt.show

##############################################################################
#line plot of suicdes per 100K over time by USSR countries
USSR_suic_trend = sns.lineplot(x='year', y='Suicides per 100K', hue ='Country', data= USSR_suic)
#move legend outside of graph
USSR_suic_trend.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#add title
USSR_suic_trend.set_title('Suicides per 100K Over Time (USSR Countries)')
#change x label to be capitalized
USSR_suic_trend.set_xlabel('Year')

##############################################################################
#line plot of suicdes per 100K over time by USSR countries
USSR_suic_whole = sns.lineplot(x='year', y='Suicides per 100K', data= USSR_whole)
#add title
USSR_suic_whole.set_title('Suicides per 100K Over Time (USSR)')
#change x label to be capitalized
USSR_suic_whole.set_xlabel('Year')

##############################################################################
#line plot of suicdes per 100K over time by continent and gender (not that pretty)
line_suic_gender = sns.lineplot(x='year', y='Suicides per 100K', hue ='Continent', style = 'sex', data= suic_gender)
#move legend outside of graph
line_suic_gender.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#add title
line_suic_gender.set_title('Suicides per 100K Over Time, by Continent and Gender')
#change x label to be capitalized
line_suic_gender.set_xlabel('Year')

##############################################################################
#line plot of Happiness Score over time by continent
line_happy_all_cont = sns.lineplot(x='year', y='Happiness Score', hue ='Continent', data= happy_all_dropna)
#x tick markers for full years
line_happy_all_cont.set(xticks = happy_all_dropna.year.values)
#move legend outside of graph
line_happy_all_cont.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#add title
line_happy_all_cont.set_title('Happiness Score Over Time, by Continent')
#change x label to be capitalized
line_happy_all_cont.set_xlabel('Year')

##############################################################################
#map for part_world (won't show in Spyder but will work in jupyter notebook)
fig = px.scatter(whole_world_avg, x="Happiness Score", y="Suicides per 100K",
                 size="gdp_md_est", color="Region", hover_name="Country",
                 log_x=True, size_max=55)
fig.show()

#map of average happiness scores from 2015-2019
fig1 = px.choropleth(happy_all_average, locations="Country_Code",
                    color="Happiness Score", # lifeExp is a column of gapminder
                    hover_name= 'Country', # column to add to hover information
                    color_continuous_scale=px.colors.sequential.Greens_r)
fig1.show()

#map of average happiness scores from 2015-2019
fig2 = px.choropleth(suic_average, locations="Country_Code",
                    color="Suicides per 100K", # lifeExp is a column of gapminder
                    hover_name= 'Country', # column to add to hover information
                    color_continuous_scale=px.colors.sequential.Greens)
fig2.show()

##############################################################################
#bubble graphs (won't work in Spyder but will work in jupyter notebook)
#filter whole_world by year and sort by Continent then Country
whole_world_2015 = whole_world[whole_world['year'] == 2015]
whole_world_2015 = whole_world_2015.sort_values(['Continent','Country'])
whole_world_2016 = whole_world[whole_world['year'] == 2016]
whole_world_2016 = whole_world_2016.sort_values(['Continent','Country'])

#create empty arrays for bubbles (bubble graphs)
bubble_size15 = []
bubble_size16 = []

#create bubbles for each country based on population (year 2015)
for index, row in whole_world_2015.iterrows():
    bubble_size15.append(round(math.sqrt(row['population']),0))
    
#create bubbles for each country based on population (year 2016)
for index, row in whole_world_2016.iterrows():
    bubble_size16.append(round(math.sqrt(row['population']),0))

#add bubble size to dataframe
whole_world_2015['bubble_size'] = bubble_size15
whole_world_2016['bubble_size'] = bubble_size16

#Create list of continent names (2015)
continent_names = ['Asia', 'South America', 'Europe', 'Africa', 'North America', 'Oceania']
continent_data15 = {continent:whole_world_2015.query('Continent == "%s"' %continent)
                 for continent in continent_names}

#Create list of continent names (2016)
continent_data16 = {continent:whole_world_2016.query('Continent == "%s"' %continent)
                 for continent in continent_names}

##############################################################################
#create bubble graph for 2015
fig15 = go.Figure()

for continent_name, continent in continent_data15.items():
    fig15.add_trace(go.Scatter(
        x=continent['Happiness Score'],
        y=continent['per_capita'],
        name=continent_name,
        text=whole_world_2015['Country'],
        hovertemplate=
        "<b>%{text}</b><br><br>" +
        "Happiness Score: %{x:,.2f}<br>" +
        "Suicides per Capita: %{y:,.5%}<br>" +
        "Population: %{marker.size:,}" +
        "<extra></extra>",
        marker_size=continent['bubble_size'],
        ))

fig15.update_traces(
    mode='markers',
    marker={'sizemode':'area',
            'sizeref':10})

fig15.update_layout(
    xaxis={
        'title':'Happiness Score',
        'type':'log'},
    yaxis={'title':'Suicides per Capita'},
    title_text = "Suicides per Capita against Happiness Score, by Population (2015)")

fig15.show()
##############################################################################

##############################################################################
#create bubble graph for 2016
fig16 = go.Figure()

for continent_name, continent in continent_data16.items():
    fig16.add_trace(go.Scatter(
        x=continent['Happiness Score'],
        y=continent['per_capita'],
        name=continent_name,
        text=whole_world_2016['Country'],
        hovertemplate=
        "<b>%{text}</b><br><br>" +
        "Happiness Score: %{x:,.2f}<br>" +
        "Suicides per Capita: %{y:,.5%}<br>" +
        "Population: %{marker.size:,}" +
        "<extra></extra>",
        marker_size=continent['bubble_size'],
        ))

fig16.update_traces(
    mode='markers',
    marker={'sizemode':'area',
            'sizeref':10})

fig16.update_layout(
    xaxis={
        'title':'Happiness Score',
        'type':'log'},
    yaxis={'title':'Suicides per Capita'},
    title_text = "Suicides per Capita against Happiness Score, by Population (2016)")

fig16.show()    
##############################################################################
