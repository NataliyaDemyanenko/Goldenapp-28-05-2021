#!/usr/bin/env python3
import math
from math import *
import matplotlib.pyplot as plt
import networkx as nx
from cooperative_game import * 
import pickle
import string
from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas as pd

def powerset(s):
     x = len(s)
     masks = [1 << i for i in range(x)]
     for i in range(1 << x):
         yield [ss for mask, ss in zip(masks, s) if i & mask]

def goldenapp():
     entries={}
     party_names={}
     polls={}

     entries = pickle.load( open( 'save.p', "rb" ) )

     s=0
     party_num=0
     for k in range(0, 15):
         try:
             polls[k]=float(entries[k+1,3])
             s=s+1
         except ValueError:
             party_num=s


     # Initialize parties and coalitions (labelled by letters)

     labels=['']*party_num
     colors=['']*party_num
     partyposition=['']*party_num
     excluded=['']*party_num


     #line - relevant excel line for party positions
     line=['']*party_num

     #policy dimensions
     rile=['']*party_num
     planeco=['']*party_num
     markeco=['']*party_num
     welfare=['']*party_num
     intpeace=['']*party_num

     #Factor analysis
     fac1=['']*party_num
     fac2=['']*party_num
     fac3=['']*party_num
     fac4=['']*party_num
     fac5=['']*party_num

     
     parties = list(string.ascii_uppercase)[0:party_num]

     for k in range(0, party_num):
         party_names[k]=str(entries[k+1,0])
         labels[k]=str(entries[k+1,1])
         colors[k]=str(entries[k+1,2])
         excluded[k]=list(str(entries[k+1,4]))
         #print(k,excluded[k])
         line[k]=int(entries[k+1,5])
         #partyposition[k]=float(entries[k+1,4])
         #print(k, line[k])
     
     #Import data from excel
     data = pd.read_excel (r'/home/nataliya/MPDatasetMPDS2020b.xlsx')
     df = pd.DataFrame(data, columns= ['rile','planeco','markeco','welfare','intpeace'])
     #print (df)

     #Select specific values for the parties
     for k in range(0, party_num):
         rile[k]=df.at[line[k]-2,'rile']
         #print(k,rile[k])
         planeco[k]=df.at[line[k]-2,'planeco']
         #print(k,planeco[k])
         markeco[k]=df.at[line[k]-2,'markeco']
         #print(k,markeco[k])
         welfare[k]=df.at[line[k]-2,'welfare']
         #print(k,welfare[k])
         intpeace[k]=df.at[line[k]-2,'intpeace']
         #print(k,intpeace[k])
     
     #Calculation for factor analysis. Version of the database: 2020b downloaded on 24.02.2021
     for k in range(0, party_num):
         fac1[k]= -0.293*rile[k] + 0*planeco[k] + 0*markeco[k] + 0.967*welfare[k] + 0*intpeace[k]
         #print(k,fac1[k])
         fac2[k]= 0.246*rile[k] + 0*planeco[k] + 0.975*markeco[k] + 0*welfare[k] + 0*intpeace[k]
         #print(k,fac2[k])
         fac3[k]= -0.23*rile[k] + 0.983*planeco[k] + 0*markeco[k] + 0*welfare[k] + 0*intpeace[k]
         #print(k,fac3[k])
         fac4[k]= -0.154*rile[k] + 0*planeco[k] + 0*markeco[k] + 0*welfare[k] + 0.991*intpeace[k]
         #print(k,fac4[k])
         fac5[k]= 0.882*rile[k] - 0.17*planeco[k] + 0.187*markeco[k] - 0.225*welfare[k] - 0.112*intpeace[k]
         #print(k,fac5[k])



     #Calculate distance between party positions
     dist={}
     for k in range(0, party_num):
         for j in range(0, party_num):
             dist[k,j] = sqrt(pow(fac1[k]-fac1[j],2)+pow(fac2[k]-fac2[j],2)+pow(fac3[k]-fac3[j],2)+pow(fac4[k]-fac4[j],2)+pow(fac5[k]-fac5[j],2))
     
     print(k, dist)

     #Maximum distance - maximum theoretical disctance
     #maxdist = max(dist.values())
     maxdist = 226.32728514255632
     print('Maximum distance:')
     print(maxdist)

     #Set threshold for compatibility
     thr = 0.3*maxdist
     print('Threshold:')
     print(thr)

     #Exclude parties with distance > threshold of max distance (threshold) 
     for k in range(0, party_num):
         for j in range(0, party_num): 
             #print(k,j)  
             if dist[k,j] > thr and k!=j and j not in excluded[k]:
                 print(k, j,dist[k,j])
                 excluded[k] += parties[j]
                 #print(k, excluded[k])

     #Add back connections with parties with unknown political position
     for k in range(0, party_num):
         if line[k] == 2:
             excluded[k].clear()
             print(k, excluded[k])

     label = dict(zip(parties,labels))  
     color = dict(zip(parties,colors))
     coalitions = powerset(parties)


     # Introduce campaign commitments

     fr={}
     friends={}
     for k in range(0,party_num):
         fr[k]=list(set(parties) - set(excluded[k]))

     for k in range(0,party_num):
         friends[parties[k]]=fr[k]
         #print(friends[parties[k]])

     # Computing seats, Shapley values and all winning coalitions

     P=0
     for i in range(len(polls)):
         P += polls[i]


     # Initialize proportions of seats (precise and rounded)    

     s ={}
     sround = {}    
     pl = {} 
     i = 0  
     for p in parties:
         pl[p]=polls[i]
         s[p] = polls[i]/P
         sround[p]= round(float(s[p]*100),1)
         i+=1

     worth = {}                                           # Assign worth to coalitions
     mworth = {}
     for i in tuple(coalitions):
         #print(i)
         sumsp=0
         for r in tuple(i):
             j=set(i).intersection(set(friends[r]))
             if j==set(i):
                 sumsp = sumsp +  s[r]
         #print(sumsp)
         worth[tuple(i)]=0
         mworth[tuple(i)]=0
         if (sumsp > 1/2):
             worth[tuple(i)] = 1
             mworth[tuple(i)] =1        
         #worth[tuple(i)] = ( copysign(1,(sumsp - 0.5)) + 1)/2
         #if ( copysign(1,(sumsp - 0.5)) + 1)==1:
             #worth[tuple(i)] = 0
         for j in tuple(powerset(i)):                       # Make game monotonic
             mworth[tuple(i)]=max(mworth[tuple(i)], mworth[tuple(j)])

     #print('Worth', mworth)
     letter_game = CooperativeGame(mworth)
     sh = letter_game.shapley_value()
     #print(sh)
     print( "{:<10} {:<10} {:<10} {:<10} {:<10}".format('Label', 'Party', 'Votes (%)', 'Seats (%)', 'Strength') )
     for k in parties:
         lb = label[k]
         num = sround[k]
         v = sh[k]
         #v = max(sh[k],0)
         print( "{:<10} {:<10} {:<10} {:<10} {:<10}".format(k, lb, round(float(pl[k]),2), num, v) )    

     letter_function = {}
     for k in worth.keys():            # Find all winning coalitions. N: this includes incompatible coalitions also
         if worth[k] != 0:
             letter_function[k]=worth[k]
     #print('Letter function', letter_function)


     # Find all minimal winning coalitions

     non_minimal_winning={}
     for k in letter_function.keys():
         for j in letter_function.keys():
             if (j!= k) and (set(k).intersection(set(j)) == set(k)):             
                 non_minimal_winning[j]=letter_function[j]

     minimal_winning={}
     for k in letter_function.keys():
         if not(k in non_minimal_winning.keys()):
             minimal_winning[k]=letter_function[k] 


     # Find all stable coalitions

     plt.figure(0)                
     chi = {}
     power = {}
     for k in minimal_winning.keys():
         S = 0
         for j in k:
             S += max(sh[j],0)
         chi[k] = minimal_winning[k]/S

         u=''
         b = 0
         for j in k:
             po=''
             pc=''
             power[j] = max(0,sh[j])*chi[k]
             if power[j]==0:
                 po='('
                 pc=')'
             u = u + po + label[j].split('/')[0] + pc + ' '  
         for i in k:
             plt.bar(u, power[i], bottom = b, color = color[i])
             b = b +power[i]
         plt.bar(u, 0.03, bottom=(chi[k]-1)*(0.9), color='white', width=.2) 
     plt.xticks(rotation=-20, fontsize=8, horizontalalignment='left')

     print('Minimal winning coalitions and Power distribution') 
     print('( Power = Strength x Stability ):')            

     S = 0
     for j in parties:
         S += max(sh[j],0)

     # Calculate stability for all winning coalitions
     chi2 = {}
     for k in letter_function.keys():
         S2 = 0
         for j in k:
             S2 += max(sh[j],0)
         chi2[k] = letter_function[k]/S2
     #print(chi2)


     #Sum of all stability coefficients
     print('Sum of all stability coefficients:')

     SChi2 = sum(chi2.values())
     print(SChi2)

     #Calculate sum of stability (SS) and new stability rank (SR)
     for k in letter_function.keys():
             for j in k:          
                SS = sum(value for key, value in chi2.items() if value >= chi2[k])
                SR = SS / SChi2
             print(k, SS, SR)

     plt.figure(1)                
     for i in parties:
         plt.bar(label[i], s[i], color = color[i], width=0.3, align='center')
         plt.bar(label[i], 0.003, bottom = max(0,sh[i])/S, color = 'red', width=0.6, align='center')         
     plt.xticks(rotation=-20, fontsize=8, horizontalalignment='left')
     
     plt.figure(2)
     G = nx.Graph()
     G.add_nodes_from(parties)
     for i in tuple(parties):
         for j in tuple(parties):
             if set(i).intersection(friends[j])!=set():
                 G.add_edge(i,j)
     pos = nx.spring_layout(G)  # positions for all nodes
     # nodes 
     deg=dict(nx.degree(G))

     nx.draw_networkx_nodes(G, pos, nodelist=parties, node_color=colors, edgecolors='black', alpha=0.5, node_size=[v * 10000 for v in s.values()])
     nx.draw_networkx_nodes(G, pos, nodelist=parties, node_color=colors, edgecolors='red', alpha=0.5, node_size=[v * 10000 for v in sh.values()])

     # edges
     nx.draw_networkx_edges(G, pos, alpha=0.2, width=1.5)
     # labels
     nx.draw_networkx_labels(G, pos, labels=label, font_size=10, font_family='sans-serif')

     plt.axis('off')
     plt.show()
     print(chi)


