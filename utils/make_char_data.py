# -*- coding: utf-8 -*-
import sys
import moji2 as moji
from PIL import Image, ImageFont, ImageDraw
import numpy
import pickle

FONTSIZE = 17
font = ImageFont.truetype("ipagp-mona.ttf", FONTSIZE)

from char_data import CharData

if __name__ == "__main__":
    cnt = 0
    map_char_data = {}
    map_id2img = {}
    map_img2id = {}


    for ch in moji.hankaku2.decode("utf-8"):
        #print ch,
        w, h = font.getsize(ch)
        imag = Image.new("L", (w, h), "#ffffff")
        draw = ImageDraw.Draw(imag)
        draw.text((0, 0), ch, font=font, fill="#000000")
        _imag = numpy.array(imag)
        imag = numpy.zeros((FONTSIZE, _imag.shape[1]), dtype=numpy.uint8) + 255
        imag[-_imag.shape[0]:, :] = _imag[-FONTSIZE:, :]

        map_char_data[cnt] =  numpy.where(imag > 200, 0, 1)
        map_id2img[cnt] = ch
        map_img2id[ch] = cnt
        #imag.save("char_data/%s.bmp"%cnt)
        cnt += 1



    char_data = CharData(map_id2img,
                         map_img2id,
                         map_char_data)

    #with open("map_char_data.pkl", "w") as f:
    with open("char_data_hankaku.pkl", "w") as f:
        pickle.dump(char_data, f)
