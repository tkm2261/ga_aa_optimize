# coding: utf-8
# http://d.hatena.ne.jp/yatt/20091217/1261045898
import sys
from PIL import Image, ImageFont, ImageDraw
import numpy
import pickle

FONTSIZE = 17
font = ImageFont.truetype("ipagp-mona.ttf", FONTSIZE)

class CharData(object):

    def __init__(self,
                 map_id2img,
                 map_img2id,
                 map_img_id_data):


        self.map_id2img = map_id2img
        self.map_img2id = map_img2id
        self.map_img_id_data = map_img_id_data
        self.map_img_id_char_avg = self._calc_map_img_id()
        self.map_size_img_id = self._make_map_size_img_id()

    def _calc_map_img_id(self):
        map_img_id_char_avg = {}
        for img_id, data in self.map_img_id_data.items():
            map_img_id_char_avg[img_id] = data.mean()

        return map_img_id_char_avg


    def _make_map_size_img_id(self):
        map_size_img_id = {}
        for img_id, data in self.map_img_id_data.items():
            size = data.shape[1]

            if size in map_size_img_id:
                map_size_img_id[size].append(img_id)
            else:
                map_size_img_id[size] = [img_id]

        return map_size_img_id

    def print_data(self, char):
        data = self.map_img_id_data[self.map_img2id[char]]

        for row in data:
            for col in row:
                print col,
            print



