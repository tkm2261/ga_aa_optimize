# -*- coding: utf-8 -*-

from ga_aa_optimize import *
from char_data import CharData
from load_data import *
from matplotlib import pyplot as plt
from matplotlib import cm
import unittest
import pickle
import time
from logging import getLogger, basicConfig
log_fmt = '%(asctime)s %(name)s.%(funcName)s %(lineno)d [%(levelname)s]%(message)s'
basicConfig(format=log_fmt,
                    filemode='w',
                    level='DEBUG')

logger = getLogger(__name__)


if __name__ == "__main__":
    with open("char_data_hankaku.pkl") as f:
    #with open("map_char_data.pkl") as f:
        char_data = pickle.load(f)

    image_matrix = load_image("test.jpg", is_edge=False)

    ga_optimize = GAAaOptimize(image_matrix,
                                         char_data,
                                         fontsize=17)
    #plt.imshow(image_matrix, cmap=cm.Greys)
    #plt.show()
    #self.char_data.print_data("_")

    t = time.time()
    image_data = ga_optimize.optimize(max_iter=10)
    print image_data.objective
    print image_data.make_text(char_data.map_id2img)
    image_data.save_data("champ.pkl")
    image_data.draw(fig_name="champ.png")


    """
    image_data = self.ga_optimize.generate_image()
    min_ovjective, list_row_objctive, row_shift = self.ga_optimize.calc_objective(image_data)
    print min_ovjective, row_shift
    image_data.show()
    image_data2 = self.ga_optimize.local_search(image_data, row_shift)
    image_data2.show()
    """