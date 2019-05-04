# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 19:17:31 2019

@author: javig
"""

import json
import pandas as pd
import numpy as np
from collections import Counter
import math
# =============================================================================
# 
# def add_to_matrix(matrix, pid, tid):
#     try:
#         matrix[pid]
#     except IndexError:
#         #No exite la playlist
#         prev_lane = matrix[pid-1]
#         length = len(prev_lane)
#         if length <= tid:
#             length += 1
#             new_column = np.zeros((pid,1), dtype=int)
#             matrix = np.hstack([matrix,new_column])
#         new_lane = np.zeros((length,), dtype=int)
#         new_lane[tid]=1
#         matrix = np.vstack([matrix,new_lane])
#     try:
#         matrix[pid][tid] = 1
#     except IndexError:
#         new_column = np.zeros((pid+1,1), dtype=int)
#         new_column[pid][0] = 1
#         matrix = np.hstack([matrix,new_column])
#     return matrix
# 
# def json_to_csv1(name_json, name_csv):
#     matrix = np.array([[0]])
#     id_tracks = {}
#     data = json.load(open(name_json))
#     
#     count =0
#     playlists = data['playlists']
#     for i in range(len(playlists)):
#         tracks = playlists[i]["tracks"]
#         pid = playlists[i]["pid"]
#         for track in tracks:
#             track_uri = track["track_uri"]
#             if id_tracks.get(track_uri) == None:
#                 id_tracks[track_uri]=count
#                 count += 1           
#             matrix = add_to_matrix(matrix,pid,id_tracks.get(track_uri))
#     df = pd.DataFrame.from_records(matrix)
#     df.to_csv(name_csv,index=False) 
# =============================================================================
    
def intersection_count(list1, list2):#Borrar
    result = 0
    for i in range(len(list1)):
        if list1[i] == list2[i] == 1:
            result += 1
    return result

def union_count(list1, list2):#BORRAR
    result = 0
    for i in range(len(list1)):
        if list1[i] == 1 or list2[i] == 1:
            result += 1
    return result

def json_to_csv(name_json, name_csv):
    matrix = []
    id_tracks = {}
    data = json.load(open(name_json))
    count = 0
    old_pid = -1
    playlists = data['playlists']
    for playlist in playlists:
        tracks = playlist["tracks"]
        pid = playlist["pid"]
        for track in tracks:
            track_uri = track["track_uri"]
            id_track = id_tracks.get(track_uri)
            if id_track == None:
                id_tracks[track_uri] = count
                id_track = count
                count += 1
            if old_pid != pid:
                matrix.append([])
            matrix[pid].append(id_track)
            old_pid = pid
        print(pid)
    df = pd.DataFrame(matrix,dtype=np.int)
    df.to_csv(name_csv,index=False)   
                

def similarity_matrix(name_csv):
    df = pd.read_csv(name_csv)
    matrix = df.values
    num_playlist = len(matrix)
    similarity = []
    old_i =-1
    for i in range(num_playlist):
        list1 = matrix[i]
        print(i)
        for j in range(num_playlist):  
            if old_i != i:
                similarity.append([])
            list2 = matrix[j]
            intersection = set(list1).intersection(list2)
            union = set(list1).union(list2)
            intersection = {x for x in intersection if pd.notna(x)}
            union = {x for x in union if pd.notna(x)}
            similarity[i].append(len(intersection) / len(union))
            old_i = i
    df = pd.DataFrame.from_records(similarity)
    df.to_csv("similarity.csv",index=False) 
    return similarity

def temp_update_similarity_and_playlists(name_csv, similarity, new_playlist):
    df = pd.read_csv(name_csv)
    simil = pd.read_csv(similarity)
    matrix = df.values
    df.loc[len(matrix)-1] = pd.Series(new_playlist)
    matrix = df.values
    num_playlist =len(matrix)
    list1 = new_playlist
    similarity = []
    for j in range(num_playlist):  
        list2 = matrix[j]
        intersection = set(list1).intersection(list2)
        union = set(list1).union(list2)
        intersection = {x for x in intersection if pd.notna(x)}
        union = {x for x in union if pd.notna(x)}
        similarity.append(len(intersection) / len(union))
    #print(simil[1000])
    simil[str(num_playlist-1)] = pd.Series(similarity)
    simil.loc[num_playlist-1] = pd.Series(similarity)
    
    return simil,df,num_playlist-1
# =============================================================================
# def predict(name_similarity_matrix):
#     K = 25
#     neighbors = []
#     df = pd.read_csv(name_similarity_matrix)
#     matrix = df.values
#     num_playlist = len(matrix)
#     for i in range(num_playlist):
#         for j in range(num_playlist):
#             pass
#     #alamcenar los k vecinos en una lista ordenada, realizar elpredict y despues el MSE, para esto tengo que volver a hacer el procesamiento de datos partiendo en 80/20
# 
# =============================================================================
def predict_for_playlsit(pid, similarity, playlists, actual_pl):#BORRAR Al TERMINAR
    #coger la fila del pid
    #seleccionar los k valores mas altos
    #contar las canciones que se repiten entre esas listas
    #devilver n canciones que ams se repitan
# =============================================================================
#     df = pd.read_csv(similarity)
#     dfs = pd.read_csv(playlists)
# =============================================================================
    df = similarity
    dfs = playlists
    matrix = dfs.values
    a = df.nlargest(6,str(pid))
    a = a[1:]
    index = a[str(pid)].index
    songs = []
    for i in index:
        p = matrix[i]
        songs = songs + get_songs_not_in_playlist(p,actual_pl)
    count = Counter(songs)
    return (count.most_common(5))

def get_songs_not_in_playlist(list1,list2):
    songs = []
    for i in list1:
        if str(i) != 'nan' and (i not in list2):
            songs.append(i)
    return songs
    
def check_any_list_in_list(list1, list2):
    print(list1)
    print(list2)
    for i in list1:
        if i[0] in list2:
            return True
    return False

def test(playlists_train, playlist_test, similarity):
    #recorrer la playlist, comprobar si alguna de las canciones del meotdo predict estan en la lista
    p=[]
    score = 0.
    for i in range(len(playlist_test)):
        p = p +[playlist_test[i]]
        simil,playlists, pid = temp_update_similarity_and_playlists(playlists_train,similarity,p)
        predict = predict_for_playlsit(pid,simil,playlists,p)
        if check_any_list_in_list(predict, playlist_test):
            score +=1
        #if playlist_test[i+1] in predict:
            #score+=1
        print("Actual list: ",p)
        print("prediction: ",predict)
        print("actual score: ",score)
        print("---")
    score = score/len(playlist_test)
    print("final score = ",str(score))

def scores(all_playlists, playlist_test, similarity):
    simil,playlists, pid = temp_update_similarity_and_playlists(all_playlists,similarity,playlist_test)
    matrix = playlists.values
    index = simil.nlargest(100,str(pid))[1:][str(pid)].index
    songs = []
    for i in index:
        songs += get_songs_not_in_playlist(matrix[i],matrix[pid])
    count = Counter(songs)
    most_common_tuple = count.most_common(500)
    songs_recommended = [song for song, cont in most_common_tuple]
    rp = r_precision(playlist_test,songs_recommended)
    ndcg = normalized_discounted_comulative_gain(playlist_test,songs_recommended)
    c = clicks(playlist_test,songs_recommended)
    return rp, ndcg, c

def r_precision(playlist_test, songs_recommended):
	#Número de canciones relevantes conseguidas dividida por el número de canciones relevantes conocidas
    lenG = len(playlist_test)
    intersection = set(playlist_test).intersection(songs_recommended[0:lenG])
    result = len(intersection)/lenG
    return result

def normalized_discounted_comulative_gain(playlist_test, songs_recommended):
    #DCG / IDCG
    #relevancia	
    rel = []
    for song in songs_recommended:
        rel.append(1) if song in playlist_test else rel.append(0)	
    dcg = rel[0]
    for i in range(1,len(songs_recommended)):
        dcg += (rel[i]/math.log(i+1,2))
    idcg = 1
    for i in range(1,len(playlist_test)):
        idcg += (1/math.log(i+1,2))
    ndcg = dcg/idcg
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
    read_and_save = False
    get_similarity_matrix = False
    testb = False
    if read_and_save:
        json_to_csv('mpd.slice.0-999.json',"pruebas3.csv")
    elif get_similarity_matrix:
        similarity = similarity_matrix("similarity.csv")
    elif testb:
        p = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,11880]
        test("pruebas3.csv",p,"similarity.csv")
    else:
        #print(predict_for_playlsit(0,"similarity.csv","pruebas3.csv"))        
        p = [9,10,11,12,13,14,15,16,17,18,19,20,11880]
        rp, ndcg,c = scores("pruebas3.csv",p,"similarity.csv")
        print("R precision: ",rp)
        print("NDCG: ",ndcg)
        print("Clicks: ",c)
    
    print("done")