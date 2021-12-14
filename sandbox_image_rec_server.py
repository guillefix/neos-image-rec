from http.server import BaseHTTPRequestHandler, HTTPServer
import re
import numpy as np
import matplotlib.pyplot as plt

from urllib.parse import urlparse
from urllib.parse import parse_qs

import re
import json
from collections import Counter
import sys

def hex_to_rgb(value):
    # value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

import tensorflow as tf
import tensorflow.keras as keras
import cv2
from PIL import Image
from clip_embeddings import embed_text, embed_image
# model = keras.applications.mobilenet_v2.MobileNetV2(input_shape=resized_image.shape, alpha=1.0, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
# model = keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
model = keras.applications.vgg19.VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
import json
class_idx = json.load(open("imagenet_class_index.json","rb"))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
from imagenet_index_to_class import index_to_class
def predict_img_label(image):
    height, width, depth = image.shape
    image = image.astype(np.uint8)
    # image = Image.fromarray(image)
    new_height = 224
    # new_height = 299
    new_width = int(width*(new_height/height))
    # print(new_width)
    resized_image = cv2.resize(image, dsize=(new_width, new_height))
    resized_image = resized_image[:,int(new_width/2-new_height/2):int(new_width/2-new_height/2)+new_height]
    pred = model.predict(np.expand_dims(resized_image,0))
    index = np.argmax(pred)
    return index_to_class[index]


def predict_img_label2(image):
    # queries = ["forest", "city", "furry", "sci-fi", "space station interior", "grass"]
    queries = ["anthro avatar", "party-goer avatar", "anime girl vrchat avatar", "anime boy vrchat avatar", "robot avatar", "avatar with tail"]
    query_embeddings = embed_text(queries)
    normalized_query_embeddings = query_embeddings / np.linalg.norm(query_embeddings,axis=1, keepdims=True)
    image = (np.array(image)* 255).astype(np.uint8)
    # temp = get_snapshots(imagepath)
    def get_clip_scores(snapshots, plot_image=False):
        image_embeddings = embed_image(snapshots, fromarray=True)
        image_embeddings = image_embeddings.T
        # if plot_image:
        #     plot_image_grid(temp,(3,3))

        dots = np.dot(normalized_query_embeddings,image_embeddings/np.linalg.norm(image_embeddings,axis=0))
        dots = np.sum(dots,axis=1)
        return list(dots), queries[np.argmax(dots)]
    scores, queries = get_clip_scores([image])
    if queries == "anime girl vrchat avatar" or queries == "anime boy vrchat avatar":
        queries = "anime"
    elif queries == "anthro avatar" or queries == "avatar with tail":
        queries = "furry"
    elif queries == "party-goer avatar":
        queries = "party-goer"
    elif queries == "robot avatar":
        queries = "robot"
    return queries

if __name__ == "__main__":
    class S(BaseHTTPRequestHandler):
        def _set_headers(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

        def do_GET(self):
            print(self.path)
            if self.path == "/favicon.ico": return
            self._set_headers()
            # query=self.path[1:]
            query_params = parse_qs(urlparse(self.path).query)
            sys.stdout.flush()
            self.wfile.write(bytes(str(results_str), "utf-8"))

        def do_HEAD(self):
            self._set_headers()

        def do_POST(self):
            # Doesn't do anything with posted data
            content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
            post_data = self.rfile.read(content_length) # <--- Gets the data itself
            self._set_headers()
            # print(post_data.decode('utf-8'))
            image_str = post_data.decode('utf-8')
            # print(image_str.split(","))
            colors = np.array([list(hex_to_rgb(x)) for x in image_str.split(",")[1:]])
            image_size = int(np.sqrt(colors.shape[0]))
            colors = colors.reshape((image_size, image_size, 3)).transpose((1,0,2))
            print(colors)
            # label = predict_img_label(colors)
            label = predict_img_label2(colors)
            print(label)
            # plt.imshow(colors)
            # plt.show()
            self.wfile.write(bytes(label, "utf-8"))

    def run(server_class=HTTPServer, handler_class=S, port=80):
        server_address = ('', port)
        httpd = server_class(server_address, handler_class)
        print('Starting httpd...')
        httpd.serve_forever()

    run(port=6969)
