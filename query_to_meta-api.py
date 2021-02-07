import argparse
import os
import json
from flask import Flask, request, jsonify, make_response

import pickle
import numpy as np
import pandas as pd

import functions as F

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.errorhandler(404)
def pageNotFound(error):
    return "page not found"

@app.errorhandler(500)
def raiseError(error):
    return error

@app.route('/')
def query_to_meta_api():
    doc = request.args['query']
    pos_tokens = [word for word in doc.split(" ") if len(word) > 1]
    model_input = list(set(pos_tokens))
    video_results, audio_results = F._query_retrieval(model_input, word_vectors, video_tags, audio_tracks, audio_meta, df_video_meta, MSD_id_to_7D_id, id_to_path)

    output = {
        'query': doc,
        'model_input' : model_input,
        'video_results': video_results,
        'audio_results': audio_results
    }

    return jsonify(**output)

if __name__ == "__main__":
    global video_tags
    global audio_tracks
    global audio_meta
    global word_vectors

    global id_to_path
    global MSD_id_to_7D_id
    global df_video_meta

    parser = argparse.ArgumentParser(description='Flask option arguments')
    parser.add_argument('--host', type=str, default=None, help='Default is localhost')
    parser.add_argument('--port', type=int, default=None, help='Default is :5000')
    args = parser.parse_args() 
    host = args.host
    port = args.port

    ## Model & Meta Data Load
    with open("./static/vectors/video_tags.pkl", 'rb') as f:
        video_tags = pickle.load(f)
    with open("./static/vectors/audio_tracks.pkl", 'rb') as f:
        audio_tracks = pickle.load(f)
    with open("./static/meta/audio_meta.pkl", 'rb') as f:
        audio_meta = pickle.load(f)
    with open("./static/vectors/word_vectors.pkl", 'rb') as f:
        word_vectors = pickle.load(f)

    id_to_path = pickle.load(open("./static/meta/7D_id_to_path.pkl",'rb'))
    MSD_id_to_7D_id = pickle.load(open("./static/meta/MSD_id_to_7D_id.pkl",'rb'))
    df_video_meta = pd.read_csv("./static/meta/video_meta.csv", index_col=0)

    print("Finish Loading Audio & Meta Data")

    app.run(host="0.0.0.0" , port=5000)
