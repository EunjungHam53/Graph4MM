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

class DataParser():
    def __init__(self, batch_start=None, batch_size=4000):
        self.path = './wikiweb2m/raw/'
        self.filepath = 'wikiweb2m-*'
        self.suffix = '.tfrecord*'
        
        # Config cho batch processing
        self.batch_start = batch_start or 50000
        self.batch_size = batch_size
        self.batch_end = self.batch_start + batch_size
        
        print(f"Processing batch: {self.batch_start} -> {self.batch_end}")
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

    def download_images_batch(self, max_size_mb=15*1024):  # 15GB limit
        """
        Download images with size monitoring
        max_size_mb: Maximum total size in MB before stopping
        """
        headers = {"User-Agent": "research (https://www.cs.cmu.edu/; minjiy@cs.cmu.edu)"}
        
        # Tạo thư mục theo batch
        image_dir = f'../images/batch_{self.batch_start}_{self.batch_end}'
        os.makedirs(image_dir, exist_ok=True)
        
        total_size = 0  # Track total downloaded size
        downloaded_count = 0
        
        for page_id, d in enumerate(self.dataset):
            # Skip pages outside batch range
            if page_id < self.batch_start:
                continue
            if page_id >= self.batch_end:
                break
                
            if page_id % 500 == 0:
                print(f"Page {page_id}, Downloaded: {downloaded_count} images, Size: {total_size/1024:.1f}MB")
                
            # Check size limit
            if total_size > max_size_mb * 1024 * 1024:
                print(f"Reached size limit {max_size_mb}MB at page {page_id}")
                break
                
            image_urls = tf.sparse.to_dense(d[1]['section_image_url']).numpy()
            
            for section_id in range(image_urls.shape[0]):
                for image_id in range(image_urls[section_id].shape[0]):
                    image_url = image_urls[section_id][image_id]
                    if image_url == b'':
                        continue
                        
                    image_url = image_url.decode()
                    file_format = os.path.splitext(image_url)[1][1:]
                    if not file_format:
                        file_format = 'jpg'
                        
                    file_name = f'{image_dir}/{page_id}_{section_id}_{image_id}.{file_format}'
                    
                    # Skip if already exists
                    if os.path.exists(file_name):
                        total_size += os.path.getsize(file_name)
                        continue

                    try:
                        response = requests.get(image_url, headers=headers, timeout=10)
                        response.raise_for_status()
                        
                        # Write and optimize image
                        with open(file_name, 'wb') as file:
                            for chunk in response.iter_content(8192):
                                file.write(chunk)
                        
                        # Optimize image to reduce size
                        try:
                            img = Image.open(file_name)
                            # Resize if too large
                            if img.size[0] > 1024 or img.size[1] > 1024:
                                img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                            
                            # Save with compression
                            if file_format.lower() in ['jpg', 'jpeg']:
                                img.save(file_name, 'JPEG', quality=85, optimize=True)
                            elif file_format.lower() == 'png':
                                img.save(file_name, 'PNG', optimize=True)
                            else:
                                img.save(file_name, format=img.format, optimize=True)
                                
                        except Exception as img_error:
                            print(f"Image processing error {file_name}: {img_error}")
                            if os.path.exists(file_name):
                                os.remove(file_name)
                            continue
                            
                        # Update size tracking
                        file_size = os.path.getsize(file_name)
                        total_size += file_size
                        downloaded_count += 1
                        
                        # Check size after each download
                        if total_size > max_size_mb * 1024 * 1024:
                            print(f"Size limit reached during download")
                            return
                            
                    except requests.exceptions.RequestException as e:
                        print(f"Download failed for {image_url}: {e}")
                        continue
                    except Exception as e:
                        print(f"Unexpected error: {e}")
                        continue
                        
                    # Small delay to avoid overwhelming servers
                    time.sleep(0.1)
        
        print(f"Batch {self.batch_start}-{self.batch_end} completed:")
        print(f"Downloaded: {downloaded_count} images")
        print(f"Total size: {total_size/1024/1024:.1f}MB")

# Usage examples for different batches
if __name__ == "__main__":
    # Batch 1: 50000-54000
    parser1 = DataParser(batch_start=50000, batch_size=4000)
    parser1.download_images_batch(max_size_mb=15*1024)  # 15GB limit
    
    # Batch 2: 54000-58000  
    # parser2 = DataParser(batch_start=54000, batch_size=4000)
    # parser2.download_images_batch(max_size_mb=15*1024)
    
    # Continue with other batches...