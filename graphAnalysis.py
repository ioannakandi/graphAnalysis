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
from networkx.algorithms.community.centrality import girvan_newman


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
print(updated_df)

#---------inserting AscCountries column (num) for country codes (str) in ascending order ------------
updated_df.insert(1, 'AscCountries', range(1, int(len(updated_df))+1))
#updated_df[['AscCountries','NumberDosesReceived']]
print(updated_df)

#similarity=pd.DataFrame(1 - squareform(pdist(updated_df.set_index('ReportingCountry'), lambda u,v: (u != v).mean())))
#print(similarity)

#correlation_df = updated_df.corr()
#print(correlation_df)

#---------calculating distance between countries (nodes) ------------
distances = pdist(updated_df[['AscCountries','NumberDosesReceived']].values, metric='euclidean')
dist_matrix = squareform(distances)
print(dist_matrix)
array_df= pd.DataFrame(dist_matrix.astype(int)) #affinity matrix
array_df.index = np.arange(1, len(array_df)+1)
array_df.columns = array_df.index
print(array_df)
array_df_stacked=array_df.stack()
array_df_stacked.to_csv('array_df.csv',index=True)
#array_df.iloc[:, 0:29].agg(lambda x: ','.join(x.values), axis=1).T
#array_df_1= array_df.iloc[:, 0:29]
print(array_df_stacked)
read_array_df= pd.read_csv('stacked_data_new.txt')
#read_array_df.loc[read_array_df['Source']==0 ,'Source']='AT'

#---------------------------------plotting the graph----------------------------------------
G=nx.from_pandas_edgelist(read_array_df, source='Source',target='Target',edge_attr='Weight')
# Plot it
#nx.draw(G, with_labels=True, node_color='pink', edge_color='#00A36C')
mapping = {1: "AT", 2: "BE", 3: "BG", 4:'CY', 5:'CZ', 6:'DE', 7:'DK', 8:'EE', 9:'EL', 10:'ES', 11:'FI', 12:'FR',
           13:'HR', 14:'HU', 15:'IE', 16:'IS', 17:'IT', 18:'LI', 19:'LT', 20:'LU', 21:'LV', 22:'NL', 23:'NO',
           24:'PL', 25:'PT', 26:'RO', 27:'SE', 28:'SI', 29:'SK' }
H = nx.relabel_nodes(G, mapping)
plt.title("Pfizer Vaccination Graph of EU Countries")
nx.draw(H, with_labels=True, node_color='pink', edge_color='#6495ED')
plt.show()

communities = girvan_newman(H)
node_groups = []
for com in next(communities):
    node_groups.append(list(com))
 
print(node_groups)
 
color_map = []
for node in H:
    if node in node_groups[0]:
        color_map.append('yellow')
    else:
        color_map.append('red')
plt.title("Communities of EU Countries")
nx.draw(H, node_color=color_map,edge_color='#6495ED', with_labels=True)
plt.show()
