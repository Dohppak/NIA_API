import argparse
import os
import json
from flask import Flask, request, jsonify, make_response
from gensim.models.keyedvectors import KeyedVectors
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
    audio_list = F.multiquery_retrieval(model.wv, model_input, search_song_indices)
    tag_list = F.multiquery_retrieval(model.wv, model_input, tag_indices)
    result_meta = [audio_meta[audio] for audio in audio_list]
    output = {
        'query': doc,
        'model_input' : model_input,
        'tag_list': tag_list,
        'file_url': [os.path.join("http://127.0.0.1:5000/static/audio_meta/audio", i) for i in audio_list],
        'result_meta' : result_meta
    }

    return jsonify(**output)

if __name__ == "__main__":
    global model
    global audio_meta
    global search_song_indices
    global tag_indices

    parser = argparse.ArgumentParser(description='Flask option arguments')
    parser.add_argument('--host', type=str, default=None, help='Default is localhost')
    parser.add_argument('--port', type=int, default=None, help='Default is :5000')
    args = parser.parse_args() 
    host = args.host
    port = args.port

    ## Model & Meta Data Load
    model = KeyedVectors.load('./static/vectors/model', mmap='r')
    with open("./static/vectors/video_tags.pkl", 'rb') as f:
        video_tags = pickle.load(f)
    with open("./static/vectors/audio_tracks.pkl", 'rb') as f:
        audio_tracks = pickle.load(f)

    print("Finish Loading Audio & Meta Data")

    app.run(host="0.0.0.0" , port=5000)
