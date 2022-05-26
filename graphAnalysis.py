import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform

#---------importing the dataset & isolating the columns we want to use-----------
old_df = pd.read_csv("data.csv", usecols=["ReportingCountry", "NumberDosesReceived", "TargetGroup", "Vaccine"])
df = old_df[old_df['TargetGroup'] == 'ALL']
df= df[df['NumberDosesReceived'] != 0 ]
df= df[df['Vaccine'] == 'COM' ]
df=df.dropna()
#print(df.isnull().sum())
#print(df)

#print(df.loc[df['ReportingCountry'] == 'AT', 'NumberDosesReceived'].sum())

#------creating a new dataframe using a list icluding unique country codes & NumberDosesReceived------
countries=df['ReportingCountry'].unique()

numberDosesReceived= []

for country in countries :
    doses=df.loc[df['ReportingCountry'] ==  country, 'NumberDosesReceived'].sum()
    numberDosesReceived.append(doses) 
     
updated_df = pd.DataFrame(list(zip(countries, numberDosesReceived)), columns =['ReportingCountry', 'NumberDosesReceived'])
#print(updated_df)

#---------inserting AscCountries column (num) for country codes (str) in ascending order ------------
updated_df.insert(1, 'AscCountries', range(0, len(updated_df)))
#updated_df[['AscCountries','NumberDosesReceived']]

#similarity=pd.DataFrame(1 - squareform(pdist(updated_df.set_index('ReportingCountry'), lambda u,v: (u != v).mean())))
#print(similarity)

#correlation_df = updated_df.corr()
#print(correlation_df)

#---------calculating nodes ------------
distances = pdist(updated_df[['AscCountries','NumberDosesReceived']].values, metric='euclidean')
dist_matrix = squareform(distances)
print(dist_matrix)