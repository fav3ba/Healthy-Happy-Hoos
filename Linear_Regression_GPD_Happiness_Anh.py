#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 13:33:49 2021

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
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot
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
#rename columns for agreement across dataframeskl
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
print(len(happy))
sorted(happy)

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

sorted(dfregion['Region'].unique())

#rename for agreement across dataframes (world df used United States of America)
#happy_all = happy_all.replace({'United States':'United States of America'})

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

#merge suicide and happy dataframes together
happy_sad = pd.merge(suic_red, happy, on=['Country','year'])

#merge world and happy_sad dataframes together
whole_world = happy_sad.copy()
whole_world = whole_world.dropna()

#whole_world df filtered for countries that have entries for 2015 and 2016
part_world = whole_world.groupby('Country').filter(lambda x: x.Country.size == 2)
part_world = part_world.dropna()
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

# *******************************************************************

#Linear Regression

# *******************************************************************
#Happiness Against Suicide
ww_happiness = whole_world['Happiness Score']
ww_happiness = np.asarray(ww_happiness )
ww_happiness = ww_happiness.reshape(-1,1)
ww_suicides = whole_world['Suicides per 100K']
ww_suicides = np.asarray(ww_suicides)
ww_suicides = ww_suicides.reshape(-1,1)

#scatter plot of suicide against happiness 
plt.scatter(ww_happiness, ww_suicides)
plt.title('Suicides per 100k Against Happiness Score')
plt.xlabel('Happiness Score')
plt.ylabel('Suicides per 100K')

#Split into 2015 and 2016 dataset
ww_happiness_2015 = whole_world_2015['Happiness Score']
ww_suicides_2015 = whole_world_2015['Suicides per 100K']
ww_happiness_2016 = whole_world_2016['Happiness Score']
ww_suicides_2016 = whole_world_2016['Suicides per 100K']

#scatter plot of suicide against happiness separate by year
plt.scatter(ww_happiness_2015, ww_suicides_2015, c='red')
plt.scatter(ww_happiness_2016, ww_suicides_2016, c='blue', marker='+')
plt.legend(["2015", "2016"])
plt.title('Suicides per 100k Against Happiness Score')
plt.xlabel('Happiness Score')
plt.ylabel('Suicides per 100K')

# originial code for linear regression

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
    
# model_happy_suicide = linear_model.LinearRegression()
# model_happy_suicide.fit(whole_world[['Happiness Score']],whole_world[['Suicides per 100K']])
# ww_suicides_pred = model_happy_suicide.predict(ww_happiness)

# print('Coefficients:', model_happy_suicide.coef_)
# print('Intercept:', model_happy_suicide.intercept_)
# print('Mean squared error (MSE): %.2f' %mean_squared_error(ww_suicides, ww_suicides_pred))
# print('R^2: %.2f' %r2_score(ww_suicides, ww_suicides_pred))

#linear regression class version to use for unit testing
class linreg:
    
    def __init__(self, database):
        self.database = database #whole_world
        self.model_happy_suicide = linear_model.LinearRegression()
        self.model_happy_suicide.fit(self.database[['Happiness Score']],self.database[['Suicides per 100K']])

    def linregcoef(self):
        return float(self.model_happy_suicide.coef_)
    # print('Coefficients:', model_happy_suicide.coef_)
    def linregintercept(self):
        return float(self.model_happy_suicide.intercept_)
    # print('Intercept:', model_happy_suicide.intercept_)

#unit test
import unittest

class LinRegTestSuite(unittest.TestCase):
    wholeworld = linreg(whole_world)
    
    def test_coef(self):
        self.assertIsInstance(self.wholeworld.linregcoef(), float)
        
    def test_int(self):
        self.assertIsInstance(self.wholeworld.linregintercept(), float)
    
if __name__ == '__main__':
    unittest.main()
    
sns.set_style('whitegrid')
sns.regplot(ww_happiness, ww_suicides)
plt.title('Suicides per 100k Against Happiness Score')
plt.xlabel('Happiness Score')
plt.ylabel('Suicides per 100K')

sns.lmplot(x='Happiness Score', y='Suicides per 100K', data=whole_world, hue='year', markers =['o', 'v'])
plt.title('Suicides per 100k Against Happiness Score')
plt.xlabel('Happiness Score')
plt.ylabel('Suicides per 100K')

#Happiness Against Trust (Government)
whole_world.columns
ww_trust = whole_world['Trust (Government Corruption)']
ww_trust = np.asarray(ww_trust)
ww_trust = ww_trust.reshape(-1,1)

sns.regplot(ww_trust, ww_suicides)
plt.title('Suicides per 100k Against Trust (Government Corruption)')
plt.xlabel('Trust (Government Corruption)')
plt.ylabel('Suicides per 100K')

sns.lmplot(x='Trust (Government Corruption)', y='Suicides per 100K', data=whole_world, hue='year', markers =['o', 'v'])
plt.title('Suicides per 100k Against Happiness Score')
plt.xlabel('Happiness Score')
plt.ylabel('Suicides per 100K')

#Happiness Aginst Economy
ww_economy = whole_world['Economy (GDP per Capita)']
ww_economy = np.asarray(ww_economy)
ww_economy = ww_economy.reshape(-1,1)

sns.regplot(ww_economy, ww_suicides)
plt.title('Suicides per 100k Against Economy (GDP per Capita)')
plt.xlabel('Economy (GDP per Capita)')
plt.ylabel('Suicides per 100K')

# *******************************************************************

#Does Money Make You Happy

# *******************************************************************

#correlation between scores and happiness
corr_df = happy_all.drop(['Country','Happiness Rank','year','Region','Continent','Country_Code','geometry'], axis=1)
corr = corr_df.corr()
sns.heatmap(corr, annot=True, cmap=sns.cm.rocket_r)

df2019.columns
happiest_countries_2019 = df2019[['Country', 'Happiness Score']].sort_values('Happiness Score', ascending = False).head(20)
saddess_countries_2019 = df2019[['Country', 'Happiness Score']].sort_values('Happiness Score', ascending = False).tail(20)

#top and botom 20 countries
sns.barplot(x=happiest_countries_2019['Country'], y=happiest_countries_2019['Happiness Score'])
plt.title('Top 20 happiest countries in the World')
plt.xticks(rotation=-90)

sns.barplot(x=saddess_countries_2019['Country'], y=saddess_countries_2019['Happiness Score'])
plt.title('Bottom 20 happiest countries in the World')
plt.xticks(rotation=-90)

#does money buy happiness
richest_2019 = df2019[['Country', 'Happiness Score','Economy (GDP per Capita)']].sort_values('Economy (GDP per Capita)', ascending = False).head(20)
sns.barplot(x=richest_2019['Country'], y=richest_2019['Happiness Score'])
plt.title('Happiness Score of The 20 Richest Countries 2019')
plt.ylim(0, 10)
plt.xticks(rotation=-90)

poorest_2019 = df2019[['Country', 'Happiness Score','Economy (GDP per Capita)']].sort_values('Economy (GDP per Capita)', ascending = False).tail(20)
sns.barplot(x=poorest_2019['Country'], y=poorest_2019['Happiness Score'])
plt.title('Happiness Score of The 20 Poorest Countries 2019')
plt.ylim(0, 10)
plt.xticks(rotation=-90)

GPD_Happy=px.scatter(data_frame = happy_all, x = 'Economy (GDP per Capita)', y = 'Happiness Score', animation_frame = 'year',
                     animation_group = 'Country', size = 'Happiness Score', color = 'Country', hover_name = 'Happiness Rank', title = 'Happiness Scores vs GDP')
plot(GPD_Happy)

happiest_countries_2015 = df2015[['Country', 'Happiness Score']].sort_values('Happiness Score', ascending = False).head(20)
happiest_countries_2016 = df2016[['Country', 'Happiness Score']].sort_values('Happiness Score', ascending = False).head(20)
happiest_countries_2017 = df2017[['Country', 'Happiness Score']].sort_values('Happiness Score', ascending = False).head(20)
happiest_countries_2018 = df2018[['Country', 'Happiness Score']].sort_values('Happiness Score', ascending = False).head(20)

#merge top 20 happiest country from 2015-2019
happiest_countries_5_years = happiest_countries_2015.append(happiest_countries_2016)
happiest_countries_5_years = happiest_countries_5_years.append(happiest_countries_2017)
happiest_countries_5_years = happiest_countries_5_years.append(happiest_countries_2018)
happiest_countries_5_years = happiest_countries_5_years.append(happiest_countries_2019)

sns.countplot(x='Country', data=happiest_countries_5_years, palette="Paired")
plt.xticks(rotation=-90)