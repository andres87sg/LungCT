from ..AbstractProducts import BaseModel
import wget
import validators
from os import path
from .Covid19Utils import generate_visual_result
import numpy as np
import tensorflow as tf
import cv2
import os
import collections


class Covid19Model(BaseModel):

    def __init__(self, weights=None, model_metadata=None):
        super().__init__(weights, model_metadata)

        self.inv_mapping = collections.OrderedDict({0: 'Normal', 1: 'Pneumonia', 2: 'COVID-19'})

        self.sess = tf.Session()
        tf.get_default_graph()
        saver = tf.train.import_meta_graph(weights)
        saver.restore(self.sess, model_metadata)
        graph = tf.get_default_graph()

        self.image_tensor = graph.get_tensor_by_name("input_1:0")
        self.pred_tensor = graph.get_tensor_by_name("dense_3/Softmax:0")

    def run_prediction(self, metadata):
        images_list = []

        for data in metadata:
            if 'url_path' in data:
                if validators.url(data['url_path']):
                    local_image_filename = wget.download(data['url_path'])
                    image = cv2.imread(local_image_filename)
                elif path.exists(data['url_path']):
                    image = cv2.imread(data['url_path'])
                else:
                    raise Exception('url_path wrong')

            private_id = ''
            if 'private_id' in data:
                private_id = data['private_id']
            image_name = data['url_path'].split('/')[-1]
            images_list.append((private_id, image, image_name))

        predictions_list = []
        for i, img in enumerate(images_list):
            image_private_id = img[0]
            image_original = img[1]
            file_name = img[2]

            image_tranformed = cv2.resize(image_original, (224, 224))
            image_tranformed = image_tranformed.astype('float32') / 255.0
            pred = self.sess.run(self.pred_tensor,
                                 feed_dict={self.image_tensor: np.expand_dims(image_tranformed, axis=0)})

            output_visual_result = generate_visual_result(prediction =dict(zip(list(self.inv_mapping.values()), pred.squeeze().tolist())), original_image = image_original, file_name = file_name)
            predictions_list.append({'private_id': image_private_id,
                                     'probability': str(round(np.max(pred), 2)),
                                     'diagnosis': self.inv_mapping[pred.argmax(axis=1)[0]],
                                     'visual_prediction': output_visual_result})

        response = {'predictions': predictions_list}

        return response

    def run_evaluation(self):
        pass

    def run_training(self):
        pass
