#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Alice Bogdan (ucn3qn), Kin Girma (zcu4kb), Anh Nguyen (hzn3kf), 
Frank Vasquez (fav3ba)
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import plotly.graph_objects as go
import plotly.express as px
import math
import geopandas as gpd

# read in spreadsheets:
path = "World Happiness Report 2015-2016/"
df2015 = pd.read_csv(path+"2015.csv")
df2016 = pd.read_csv(path+"2016.csv")
df2017 = pd.read_csv(path+"2017.csv")
df2018 = pd.read_csv(path+"2018.csv")
df2019 = pd.read_csv(path+"2019.csv")
dfsuicide = pd.read_csv("who_suicide_statistics.csv")
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

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
world = world.drop(['pop_est','gdp_md_est'], axis = 1)

#clean happiness data
# add year column to happiness data
df2015['year'] = 2015
print(len(df2015))
df2016['year'] = 2016
print(len(df2016))
df2017['year'] = 2017
print(len(df2017))
df2018['year'] = 2018
print(len(df2018))
df2019['year'] = 2019
print(len(df2019))

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
print(len(happy))
sorted(happy)

#combine all happiness data (2015-2019)
happy_all = happy.append(df2017)
happy_all = happy_all.append(df2018)
happy_all = happy_all.append(df2019)

#rename for agreement across dataframes (world df used United States of America)
happy_all = happy_all.replace({'United States':'United States of America'})

#clean suicide data
#drop nan values
suic_fil = dfsuicide.dropna()

#renaming for agreement across dataframes
suic_fil = suic_fil.rename(columns = {'country':"Country"})

# drop sex and age
suic_fil = suic_fil.drop(['sex','age'], axis=1)
# sum population and suicides per country per year
suic_red = suic_fil.groupby(['Country', 'year'])['suicides_no','population'].agg('sum')

# add per capita column
suic_red['per_capita'] = suic_red.suicides_no/suic_red.population

#merge suicide and happy dataframes together
happy_sad = pd.merge(suic_red, happy, on=['Country','year'])

#merge world and happy_sad dataframes together
whole_world = world.merge(happy_sad, on ='Country')

#whole_world df filtered for countries that have entries for 2015 and 2016
part_world = whole_world.groupby('Country').filter(lambda x: x.Country.size == 2)
sorted(part_world.Country.unique())

#outer joing for world and happy_sad dafaframes
#whole_world2 = pd.merge(happy_sad, world, how = "outer", on = 'Country')
#sorted(world['Country'].unique())
#sorted(happy_sad['Country'].unique())

#map for part_world (won't show in Spyder but will work in jupyter notebook)
fig = px.choropleth(part_world, locations="Country_Code",
                    color="Happiness Score", # lifeExp is a column of gapminder
                    hover_name= 'Country', # column to add to hover information
                    color_continuous_scale=px.colors.sequential.Plasma)
fig.show()

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


