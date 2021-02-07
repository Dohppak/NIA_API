import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def _msdid_to_audio_path(msd_id, MSD_id_to_7D_id, id_to_path):
    track_path = "./static/audio"
    fname = os.path.join(track_path, id_to_path[MSD_id_to_7D_id[msd_id]])
    return fname

def _tag_to_video_filepath(tag, df_video_meta):
    df_filtered = df_video_meta[df_video_meta['categories'] == tag]
    df_sort = df_filtered.sort_values(by="probs",ascending=False)
    return list(df_sort['fpath'])

def _get_most_similar(query_vector, target_dict):
    target_npy = np.array(list(target_dict.values()))
    df_cos = pd.DataFrame(cosine_similarity(target_npy,query_vector.reshape(1, -1)), index=target_dict.keys(), columns=["query"])
    return list(df_cos["query"].sort_values(ascending=False)[:6].index)

def _query_retrieval(query, word_vectors, video_tags, audio_tracks, audio_meta, df_video_meta, MSD_id_to_7D_id, id_to_path):
    query_vector = []
    for word in query:
        if word in word_vectors.keys():
            query_vector.append(word_vectors[word])
    if len(query_vector) == 0:
        print("Error 처리가 필요한 케이스 입니다!")
    query_vector = np.array(query_vector).mean(axis=0)
    video_tags = _get_most_similar(query_vector, video_tags)
    msd_ids = _get_most_similar(query_vector, audio_tracks)
    
    video_results = {}
    for tag in video_tags:
        video_results[tag] = _tag_to_video_filepath(tag, df_video_meta) 
    
    audio_results = {}
    for msd_id in msd_ids:
        audio_results[msd_id] = {
            "audio_path" : _msdid_to_audio_path(msd_id, MSD_id_to_7D_id, id_to_path),
            "audio_meta" : audio_meta[msd_id]
        }
    
    return video_results, audio_results