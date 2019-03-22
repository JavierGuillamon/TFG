# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 19:17:31 2019

@author: javig
"""

import json
import pandas as pd
import numpy as np
from collections import Counter
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
    
def intersection_count(list1, list2):
    result = 0
    for i in range(len(list1)):
        if list1[i] == list2[i] == 1:
            result += 1
    return result

def union_count(list1, list2):
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
    simil = pd.read_csv(similarity).values.tolist()
    matrix = df.values.tolist() 
    matrix.append(new_playlist)
    num_playlist = len(matrix)
    list1 = new_playlist
    for j in range(num_playlist):  
        simil.append([])
        list2 = matrix[j]
        intersection = set(list1).intersection(list2)
        union = set(list1).union(list2)
        intersection = {x for x in intersection if pd.notna(x)}
        union = {x for x in union if pd.notna(x)}
        simil[num_playlist-1].append(len(intersection) / len(union))
    print(len(simil))
    simildf = pd.DataFrame.from_records(similarity)
    matrixdf = pd.DataFrame.from_records(matrix)
    return simildf,matrixdf,num_playlist-1
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
def predict_for_playlsit(pid, similarity, playlists):
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
        songs = songs + [p for p in p if str(p) != 'nan' and p not in matrix[pid]]
    count = Counter(songs)
    return (count.most_common(5))
   
def test(playlists_train, playlist_test, similarity):
    #recorrer la playlist, comprobar si alguna de las canciones del meotdo predict estan en la lista
    p=[]
    score = 0.
    for i in range(len(playlist_test)):
        p = p +[playlist_test[i]]
        simil,playlists, pid = temp_update_similarity_and_playlists(playlists_train,similarity,p)
        predict = predict_for_playlsit(pid,simil,playlists)
        if(predict in playlist_test):
            score +=1
        print(p)
        print(predict)
    score = score/len(playlist_test)
    print("score = "+str(score))

     

if __name__ == '__main__':
    read_and_save = False
    get_similarity_matrix = False
    testb = True
    if read_and_save:
        json_to_csv('mpd.slice.0-999.json',"pruebas3.csv")
    elif get_similarity_matrix:
        similarity = similarity_matrix("similarity.csv")
    elif testb:
        p = [0,1,4,6,7,8,10,15,14,36,60,34,56,50,39]
        test("pruebas3.csv",p,"similarity.csv")
    else:
        print(predict_for_playlsit(0,"similarity.csv","pruebas3.csv"))
    
    print("done")