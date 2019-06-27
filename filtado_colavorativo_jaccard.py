# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 19:17:31 2019

@author: javig
"""
from timer import timeit

import json
import pandas as pd
import numpy as np
from collections import Counter
import math
from os import listdir
import os.path
import string
from scipy.spatial.distance import pdist
from collections import defaultdict
    
import pickle
from datasketch import MinHashLSHForest, MinHash


@timeit(repeat=1, number=1)
def data_preprocess(name_json,name_out):
    matrix = []
    data = json.load(open(name_json))
    playlists = data['playlists']
    for playlist in playlists:
        tList= []
        for track in playlist['tracks']:
                tList.append(track['track_uri'])
        matrix.append(tList)        
    pickle_matrix = open(name_out,"wb")
    pickle.dump(matrix, pickle_matrix)
    pickle_matrix.close()
 
@timeit(repeat=1, number=1)   
def simil3(name_pickle, forest=None, i =0):
    pickle_in = open(name_pickle,"rb")
    matrix = pickle.load(pickle_in)
    data_list ={}
    for l in matrix:   
        data_list["{0}".format(i)]=MinHash(num_perm=128)
        for d in l:
            data_list["{0}".format(i)].update(d.encode('utf8'))
        i+=1       
# =============================================================================
#     pickle_data_out = open(name_data_out,"wb")
#     pickle.dump(data_list, pickle_data_out)
#     pickle_data_out.close() 
# =============================================================================
    if forest == None:
        forest = MinHashLSHForest(num_perm=128)
    
    for k, m in data_list.items():
        forest.add(k,m)
    return forest,i
# =============================================================================
#     forest.index()
#     pickle_simil_out = open(name_simil_out,"wb")
#     pickle.dump(forest, pickle_simil_out)
#     pickle_simil_out.close()
# =============================================================================
    
@timeit(repeat=1, number=1)    
def similar_to_playlist(playlist_num,Kn,forest,data): #id de la playlist y k listas mas parecidas a devolver
    pickle_data_in = open(data,"rb")
    data_list = pickle.load(pickle_data_in)
    return (forest.query(data_list[str(playlist_num)],Kn) if (str(playlist_num) in data_list.keys()) else None)


def temp_update_similarity_forest(playlist, forest,cont):
    m = MinHash(num_perm=128)
    for d in playlist:
        m.update(d.encode('utf8'))
    forest.add(1000000+cont,m)  
    forest.index()
    return forest, m

def get_songs_not_in_playlist(list1,list2):
    songs = []
    for i in list1:
        if str(i) != 'nan' and (i not in list2):
            songs.append(i)
    return songs

def add_scores(playlist_test, forest, knn, test_size,cont): 
    forest, minhash = temp_update_similarity_forest(playlist_test[:test_size],forest,cont)
    index = forest.query(minhash,knn)
    if 1000000+cont in index:
        index = forest.query(minhash,knn+1)
        index.remove(1000000+cont)
    songs = []
    for i in index:
        if(int(i)<1000000): picklenum=str(i)[:3]
        if(int(i)<100000): picklenum=str(i)[:2]
        elif(int(i)<10000): picklenum=str(i)[:1]
        elif(int(i)<1000): picklenum=str(0)
        #print(i," ",picklenum, " ",str(i[-3:]))
        try:
            pickle_in = open("processedData/matrixTodo"+picklenum+".plk","rb")
            playlist = pickle.load(pickle_in)
            songs += get_songs_not_in_playlist(playlist[int(str(i)[-3:])],playlist_test[:test_size])
        except:
            pass
    count = Counter(songs)
    most_common_tuple = count.most_common(500)   
    songs_recommended = [song for song, cont in most_common_tuple]
    rp = r_precision(playlist_test,songs_recommended)
    ndcg = normalized_discounted_comulative_gain(playlist_test,songs_recommended)
    c = clicks(playlist_test,songs_recommended)
    return rp, ndcg, c   
         
def scoresChallenge(playlists_test, forest, knn):
    
    #1000 listas con 1 cancion
    all_rp = 0
    all_ndcg = 0
    all_c = 0
    pickle_test1 = open(playlists_test[1],"rb")
    playlists_test1 = pickle.load(pickle_test1)
    cont =0
# =============================================================================
#     for plt1 in playlists_test1:
#         rp, ndcg, c = add_scores(plt1,forest, knn, 1, cont)
#         cont+=1
#         all_rp += rp
#         all_ndcg += ndcg
#         all_c += c
#     print("All 1 cancion rp: ",all_rp/1000," ndcg: ",all_ndcg/1000," c: ",all_c/1000)
# =============================================================================
    #1000 listas 5 primeras y nombre de la lista
    all_rp = 0
    all_ndcg = 0
    all_c = 0
    pickle_test2 = open(playlists_test[2],"rb")
    playlists_test2 = pickle.load(pickle_test2)
    for plt2 in playlists_test2:
        rp, ndcg, c = add_scores(plt2,forest, knn, 5, cont)
        cont+=1
        all_rp += rp
        all_ndcg += ndcg
        all_c += c
    print("All 5 cancion rp: ",all_rp/1000," ndcg: ",all_ndcg/1000," c: ",all_c/1000)
    #1000 listas 5 primeras
    #1000 listas 10 primeras y nombre
    #1000 listas 10 primeras
    #1000 listas 25 primeras y nombre
    all_rp = 0
    all_ndcg = 0
    all_c = 0
    pickle_test3 = open(playlists_test[3],"rb")
    playlists_test3 = pickle.load(pickle_test3)
    for plt3 in playlists_test3:
        rp, ndcg, c = add_scores(plt3,forest, knn, 25, cont)
        cont+=1
        all_rp += rp
        all_ndcg += ndcg
        all_c += c
    print("All 25 cancion rp: ",all_rp/1000," ndcg: ",all_ndcg/1000," c: ",all_c/1000)
    #1000 listas 25 aleatorias
    #1000 listas 100 primeras y nombre
    all_rp = 0
    all_ndcg = 0
    all_c = 0
    pickle_test4 = open(playlists_test[4],"rb")
    playlists_test4 = pickle.load(pickle_test4)
    for plt4 in playlists_test4:
        rp, ndcg, c = add_scores(plt4,forest, knn, 100, cont)
        cont+=1
        all_rp += rp
        all_ndcg += ndcg
        all_c += c
    print("All 100 cancion rp: ",all_rp/1000," ndcg: ",all_ndcg/1000," c: ",all_c/1000)
    #1000 listas 100 aleatorias


def r_precision(playlist_test, songs_recommended):
	#Número de canciones relevantes conseguidas dividida por el número de canciones relevantes conocidas
    try:
        lenG = len(playlist_test)
        intersection = set(playlist_test).intersection(songs_recommended[0:lenG])
        result = len(intersection)/lenG
    except:
        result = 0
    return result

def normalized_discounted_comulative_gain(playlist_test, songs_recommended):
    #DCG / IDCG
    #relevancia	
    rel = []
    for song in songs_recommended:
        rel.append(1) if song in playlist_test else rel.append(0)	
    try:    
        dcg = rel[0]
        for i in range(1,len(songs_recommended)):
            dcg += (rel[i]/math.log(i+1,2))
        idcg = 1
        for i in range(1,len(playlist_test)):
            idcg += (1/math.log(i+1,2))
        ndcg = dcg/idcg
    except:
        ndcg =0
    return ndcg

def clicks(playlist_test, songs_recommended):
    cont=0
    p_set = set(playlist_test)
    for i in range(50):
        for j in songs_recommended[cont:cont+10]:
            if j in p_set:
                return i
        cont += 10
    return 51

if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    
# =============================================================================
#     jsons = {os.path.join(here+"/json/data",x) for x in os.listdir(here+"/json/data")}
#     i =0
#     for nj in jsons:
#         print(i)
#         data_preprocess(nj,"processedData/matrixTodo"+str(i)+".plk")
#         i+=1
# =============================================================================
    
# =============================================================================
#     forest=MinHashLSHForest(num_perm=128)
#     cont=0
#     for i in range(990):
#         print(i)
#         forest,cont = simil3("processedData/matrixTodo"+str(i)+".plk",forest,cont)
#     forest.index()
#     pickle_simil_out = open("simil/similmatrix990.plk","wb")
#     pickle.dump(forest, pickle_simil_out)
#     pickle_simil_out.close()
# =============================================================================
    
    
# =============================================================================
#     print("Get matrix pickle")
#     pickle_in = open("processedData/matrixTodo0.plk","rb")
#     all_playlists = pickle.load(pickle_in)
#     print("Get forest pickle")
#     forest_in = open("simil/similmatrixTodo.plk","rb")
#     forest = pickle.load(forest_in)
#     print("Start scores")
#     rp, ndcg,c =add_scores(all_playlists,all_playlists[6],forest, 100)
#     print("R precision: ",rp)  
#     print("NDCG: ",ndcg)
#     print("Clicks: ",c) 
# =============================================================================
    test_playlists=[]
    for i in range(10):
        test_playlists.append(str("processedData/matrixTodo"+str(990+i)+".plk"))
    print(test_playlists)
    print("Get forest pickle")
    forest_in = open("simil/similmatrix990.plk","rb")
    forest = pickle.load(forest_in)
    print("Scores challenge")
    scoresChallenge(test_playlists,forest,100)
    
    print("done")