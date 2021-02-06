import os
import json


def _get_w2v_most_similar(query, target_dict, word_model):
    query_vector = np.array(word_model.wv[query]).reshape(1, -1)
    target_npy = np.array(list(target_dict.values()))
    df_cos = pd.DataFrame(cosine_similarity(target_npy,query_vector.reshape(1, -1)), index=target_dict.keys(), columns=[query])
    top_10 = list(df_cos[query].sort_values(ascending=False)[:6].index)
    return top_10
