import json
import time
import glob
import pickle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from PIL import Image

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf
from collections import defaultdict

import requests

def convert_tf_to_scipy(tf_sparse_tensor):
    array = tf.sparse.to_dense(tf_sparse_tensor).numpy()
    if len(array.shape) == 2:
        array = array.reshape(-1)
    return array.tolist()

def convert_to_scipy(page_id, d):
    page_url = d[0]['page_url'].numpy()
    page_title = d[0]['page_title'].numpy()
    page_description = d[0]['clean_page_description'].numpy()
    section_title = convert_tf_to_scipy(d[1]['section_title'])
    section_depth = convert_tf_to_scipy(d[1]['section_depth'])
    section_heading = convert_tf_to_scipy(d[1]['section_heading_level'])
    section_parent_index = convert_tf_to_scipy(d[1]['section_parent_index'])
    section_summary = convert_tf_to_scipy(d[1]['section_clean_1st_sentence'])
    section_rest_sentence = convert_tf_to_scipy(d[1]['section_rest_sentence'])
    image_url =  convert_tf_to_scipy(d[1]['section_image_url'])
    image_caption = convert_tf_to_scipy(d[1]['section_image_captions'])

    return [page_id, page_url, page_title, page_description, section_title, section_depth, section_heading, \
                section_parent_index, section_summary, section_rest_sentence, image_url, image_caption]

class DataParser():
    def __init__(self, start_page=50000, end_page=100000):
        self.path = './wikiweb2m/raw/'
        self.filepath = 'wikiweb2m-*'
        self.suffix = '.tfrecord*'
        self.start_page = start_page
        self.end_page = end_page
        print(f"Processing pages: {start_page} -> {end_page}")
        self.parse_data()

    def parse_data(self):
        context_feature_description = {
            'split': tf.io.FixedLenFeature([], dtype=tf.string),
            'page_title': tf.io.FixedLenFeature([], dtype=tf.string),
            'page_url': tf.io.FixedLenFeature([], dtype=tf.string),
            'clean_page_description': tf.io.FixedLenFeature([], dtype=tf.string),
            'raw_page_description': tf.io.FixedLenFeature([], dtype=tf.string),
            'is_page_description_sample': tf.io.FixedLenFeature([], dtype=tf.int64),
            'page_contains_images': tf.io.FixedLenFeature([], dtype=tf.int64),
            'page_content_sections_without_table_list': tf.io.FixedLenFeature([] , dtype=tf.int64)
        }

        sequence_feature_description = {
            'is_section_summarization_sample': tf.io.VarLenFeature(dtype=tf.int64),
            'section_title': tf.io.VarLenFeature(dtype=tf.string),
            'section_index': tf.io.VarLenFeature(dtype=tf.int64),
            'section_depth': tf.io.VarLenFeature(dtype=tf.int64),
            'section_heading_level': tf.io.VarLenFeature(dtype=tf.int64),
            'section_subsection_index': tf.io.VarLenFeature(dtype=tf.int64),
            'section_parent_index': tf.io.VarLenFeature(dtype=tf.int64),
            'section_text': tf.io.VarLenFeature(dtype=tf.string),
            'section_clean_1st_sentence': tf.io.VarLenFeature(dtype=tf.string),
            'section_raw_1st_sentence': tf.io.VarLenFeature(dtype=tf.string),
            'section_rest_sentence': tf.io.VarLenFeature(dtype=tf.string),
            'is_image_caption_sample': tf.io.VarLenFeature(dtype=tf.int64),
            'section_image_url': tf.io.VarLenFeature(dtype=tf.string),
            'section_image_mime_type': tf.io.VarLenFeature(dtype=tf.string),
            'section_image_width': tf.io.VarLenFeature(dtype=tf.int64),
            'section_image_height': tf.io.VarLenFeature(dtype=tf.int64),
            'section_image_in_wit': tf.io.VarLenFeature(dtype=tf.int64),
            'section_contains_table_or_list': tf.io.VarLenFeature(dtype=tf.int64),
            'section_image_captions': tf.io.VarLenFeature(dtype=tf.string),
            'section_image_alt_text': tf.io.VarLenFeature(dtype=tf.string),
            'section_image_raw_attr_desc': tf.io.VarLenFeature(dtype=tf.string),
            'section_image_clean_attr_desc': tf.io.VarLenFeature(dtype=tf.string),
            'section_image_raw_ref_desc': tf.io.VarLenFeature(dtype=tf.string),
            'section_image_clean_ref_desc': tf.io.VarLenFeature(dtype=tf.string),
            'section_contains_images': tf.io.VarLenFeature(dtype=tf.int64)
        }

        def _parse_function(example_proto):
            return tf.io.parse_single_sequence_example(example_proto,
                                                        context_feature_description,
                                                        sequence_feature_description)

        data_path = glob.glob(self.path + self.filepath + self.suffix)
        raw_dataset = tf.data.TFRecordDataset(data_path, compression_type='GZIP')
        self.dataset = raw_dataset.map(_parse_function)

    def download_images(self):
        headers = {"User-Agent": "research (https://www.cs.cmu.edu/; minjiy@cs.cmu.edu)"}

        image_dir = './images'
        os.makedirs(image_dir, exist_ok=True)

        image_count = 0  # Đếm số image đã tải

        for page_id, d in enumerate(self.dataset):
            if page_id < self.start_page:
                continue
            if page_id == self.end_page:
                break
            if page_id % 1000 == 0:
                print(f'{page_id} pages processed, {image_count} images downloaded...')
                
            image_urls = tf.sparse.to_dense(d[1]['section_image_url']).numpy()
            for section_id in range(image_urls.shape[0]):
                for image_id in range(image_urls[section_id].shape[0]):
                    image_url = image_urls[section_id][image_id]
                    if image_url == b'':
                        continue
                    image_url = image_url.decode()
                    file_format = os.path.splitext(image_url)[1][1:]
                    file_name = f'{image_dir}/{page_id}_{section_id}_{image_id}.{file_format}'
                    
                    if os.path.exists(file_name):
                        image_count += 1  # Đếm file đã tồn tại
                        break

                    another_image = False
                    try:
                        response = requests.get(image_url, headers=headers, timeout=10)
                        response.raise_for_status()
                    except requests.exceptions.HTTPError as e:
                        if "404 Client Error: Not Found for url" in str(e):
                            another_image = True
                            continue
                        else:
                            time.sleep(1)
                            response = requests.get(image_url)

                    with open(file_name, 'wb') as file:
                        for chunk in response.iter_content(8192):
                            file.write(chunk)

                    try:
                        img = Image.open(file_name)
                        image_count += 1  # Đếm image tải thành công
                    except:
                        if os.path.exists(file_name):
                            os.remove(file_name)
                        another_image = True
                        continue

                    if another_image == False:
                        break

        print(f'Batch {self.start_page}-{self.end_page} completed: {image_count} images downloaded')


if __name__ == "__main__":
    # Chạy batch cụ thể
    parser = DataParser(start_page=50000, end_page=54000)
    parser.download_images()
    
    # Các batch khác:
    # parser = DataParser(start_page=54000, end_page=58000)
    # parser = DataParser(start_page=58000, end_page=62000)