import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
import networkx as nx

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
print(countries)

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

#---------calculating distance between countries (nodes) ------------
distances = pdist(updated_df[['AscCountries','NumberDosesReceived']].values, metric='euclidean')
dist_matrix = squareform(distances)
array_df= pd.DataFrame(dist_matrix.astype(int)) #affinity matrix
array_df_stacked=array_df.stack()
array_df_stacked.to_csv('array_df.csv',index=True)
#array_df.iloc[:, 0:29].agg(lambda x: ','.join(x.values), axis=1).T
#array_df_1= array_df.iloc[:, 0:29]
print(array_df_stacked)
read_array_df= pd.read_csv('stacked_data.txt')
#read_array_df.loc[read_array_df['Source']==0 ,'Source']='AT'

#---------------------------------plotting the graph----------------------------------------
G=nx.from_pandas_edgelist(read_array_df, source='Source',target='Target',edge_attr='Weight')
# Plot it
#nx.draw(G, with_labels=True, node_color='pink', edge_color='#00A36C')
mapping = {0: "AT", 1: "BE", 2: "BG", 3:'CY', 4:'CZ', 5:'DE', 6:'DK', 7:'EE', 8:'EL', 9:'ES', 10:'FI', 11:'FR',
           12:'HR', 13:'HU', 14:'IE', 15:'IS', 16:'IT', 17:'LI', 18:'LT', 19:'LU', 20:'LV', 21:'NL', 22:'NO',
           23:'PL', 24:'PT', 25:'RO', 26:'SE', 27:'SI', 28:'SK' }
H = nx.relabel_nodes(G, mapping)
nx.draw(H, with_labels=True, node_color='pink', edge_color='#6495ED')
plt.show()
