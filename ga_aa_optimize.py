# coding: utf-8
import random
import numpy
import matplotlib
from PIL import Image

import pickle
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict
from row_localsearch import *
from multiprocessing import Pool

from logging import getLogger
logger = getLogger(__name__)

SMALL_SPACE_INDEX = 0
SMALL_SPACE_WIDTH = 5
LARGE_SPACE_INDEX = 1
LARGE_SPACE_WIDTH = 5

DOT_INDEX = 3
I_INDEX = 4

LARGE_SPACE_RATE = 0.3
WEIGHT = 3.
random.seed(0)
numpy.random.seed(1)
import time

class GAAaOptimize(object):

    def __init__(self,
                 image_matrix,
                 char_data,
                 fontsize=17):


        self.image_matrix = image_matrix
        self.char_data = char_data
        self.fontsize = fontsize
        self.row_num = image_matrix.shape[0] / self.fontsize - 1
        self.col_num = image_matrix.shape[1]

        self.list_current_images = []

        self._cache_cell_obj = {}

    def optimize(self, image_num=20, max_iter=10000, folder_path="result/"):
        logger.info("start optimize with image_num=%s, max_iter=%s"%(image_num, max_iter))
        logger.info("start to make initial solutions.")
        self.list_current_images = []

        start_time = time.time()
        p = Pool()
        result = []
        for i in xrange(image_num):
            proc = p.apply_async(toapply, (self, '_make_generate_local_search_image',))
            result.append(proc)

        self.list_current_images = [proc.get() for proc in result]
        p.close()
        p.join()

        self.list_current_images = sorted(self.list_current_images,
                                          key=lambda x:x.objective)
        row_shift = self.list_current_images[0].row_shift
        logger.info("row_shift is %s."%row_shift)
        logger.info("end making initial solutions in %s seconds."%(time.time() - start_time))

        self.list_current_images = [data for data in self.list_current_images
                                    if data.row_shift==row_shift]

        best_image = self._make_best_image(self.list_current_images, row_shift)
        self.list_current_images = [best_image]# + self.list_current_images
        #plt.ion()

        for itr in xrange(max_iter):
            start_time = time.time()
            appearance_rate = numpy.array([data.objective for data in self.list_current_images])
            appearance_rate = appearance_rate/appearance_rate.sum()

            children = []

            while len(children) < image_num:
                parent1, parent2 = numpy.random.choice(self.list_current_images,
                                                       size=2,
                                                       p=appearance_rate,
                                                       replace=True)
                child = self._make_child(parent1, parent2)
                children.append(child)
             
            for i in xrange(5):
                children.append(self._make_generate_local_search_image())
           
            
            self.list_current_images = [best_image] + children
            self.list_current_images = sorted(self.list_current_images,
                                              key=lambda x:x.objective)[:image_num-1]

            best_image = self._make_best_image(self.list_current_images, row_shift)
            best_image.draw(label="%s itr %s"%(itr+1, best_image.objective), fig_name="no1.png")
            #children[0].draw(label="%s itr"%(itr+1))
            
            show_titles = ["%s itr %s"%(itr+1, best_image.objective)]
            show_children = []
            for i in xrange(3):
                self.list_current_images[i].draw(label="%s itr %s"%(itr+1, self.list_current_images[i].objective), fig_name="child%s.png"%i)
                show_children.append(self.list_current_images[i])
                show_titles.append("%s itr %s"%(itr+1, self.list_current_images[i].objective))
                
            self._make_image(best_image, show_children,
                             show_titles,
                             fig_name=folder_path+"anim_%.3d.png"%itr)
            
            self.list_current_images = [best_image] + self.list_current_images
            logger.info("%s: %s | %s"%(itr+1, best_image.objective, time.time() - start_time))

        return best_image

    def _make_image(self, champion, children, titles, fig_name=""):
        plt.close('all')
        plt.figure(figsize=(12, 8), facecolor="1")
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)
        ax2 = plt.subplot2grid((3, 3), (0, 2))
        ax3 = plt.subplot2grid((3, 3), (1, 2))
        ax4 = plt.subplot2grid((3, 3), (2, 2))

        images = [ax1, ax2, ax3, ax4]
        
        images[0].imshow(champion.image_matrix, cmap=cm.Greys, aspect='equal')
        images[0].text(.5, .95, titles[0], horizontalalignment='center',
                    transform=images[0].transAxes, fontsize=20)
        
        for i in xrange(1, len(children)+1):
            images[i].imshow(children[i-1].image_matrix, cmap=cm.Greys, aspect='equal')
            images[i].text(.5, .9, titles[i-1],
                    horizontalalignment='center',
                    transform=images[i].transAxes)
                    
        for i, img in enumerate(images):
            img.tick_params(labelbottom='off')
            img.tick_params(labelleft='off')
            img.tick_params(length=0)


            
        plt.subplots_adjust(left=0., bottom=0., right=1., top=1., wspace = 0., hspace= 0.)
        #plt.show()
        plt.savefig(fig_name)

    def _make_child(self, parent1, parent2):
        image1, image2 = parent1.image, parent1.image

        row_num = len(image1)
        child_image = []
        child_image_ids = []
        for i in xrange(row_num):
            child_row_image = []
            child_row_image_ids = []
            thresh_col = numpy.random.choice(self.col_num, size=1)
            
            col_cnt = 0
            
            while numpy.random.random() < 0.3:
                img = self.char_data.map_img_id_data[DOT_INDEX]
                child_row_image.append(img)
                child_row_image_ids.append(CharCell(DOT_INDEX,
                                              i,
                                              (col_cnt, col_cnt + img.shape[1]),
                                              img.shape[1]
                                             )
                                     )
                col_cnt += img.shape[1]
                
            for j, char_data in enumerate(parent1.image_ids[i]):
                child_row_image.append(parent1.image[i][j])
                col_cnt += parent1.image[i][j].shape[1]
                child_row_image_ids.append(char_data)
                if col_cnt > thresh_col:
                    parent1_end = col_cnt
                    break

            for j, char_data in enumerate(parent2.image_ids[i]):
                if char_data.char_cell[1] > parent1_end:
                    char_data.char_cell = (col_cnt,
                                           col_cnt + parent2.image[i][j].shape[1])
                    col_cnt += parent2.image[i][j].shape[1]
                    child_row_image.append(parent2.image[i][j])
                    child_row_image_ids.append(char_data)
                    if col_cnt > self.col_num:
                        break

            while col_cnt < self.col_num:
                img = self.char_data.map_img_id_data[LARGE_SPACE_INDEX]
                child_row_image.append(img)
                child_row_image_ids.append(CharCell(LARGE_SPACE_INDEX,
                                              i,
                                              (col_cnt, col_cnt + img.shape[1]),
                                              img.shape[1]
                                             )
                                     )
                col_cnt += img.shape[1]
            child_image.append(child_row_image)
            child_image_ids.append(child_row_image_ids)



        child = ImageData(child_image, child_image_ids, self.col_num)

        child = self.local_search(child, parent1.row_shift)
        min_objective, list_row_objective, row_shift = self.calc_objective(child, parent1.row_shift)

        child.objective = min_objective
        child.row_shift = row_shift
        child.list_row_objective = list_row_objective

        return child

    def _make_generate_local_search_image(self):
        image_data = self.generate_image()
        min_objective, list_row_objective, row_shift = self.calc_objective(image_data)
        image_data = self.local_search(image_data, row_shift)
        min_objective, list_row_objective, row_shift = self.calc_objective(image_data, row_shift)

        image_data.objective = min_objective
        image_data.row_shift = row_shift
        image_data.list_row_objective = list_row_objective

        return image_data

    def _make_best_image(self, list_images, row_shift):
        image = []
        image_ids = []
        list_row_objective = []
        for row in xrange(self.row_num):
            list_images = sorted(list_images, key=lambda x:x.list_row_objective[row])
            image.append(list_images[0].image[row])
            image_ids.append(list_images[0].image_ids[row])
            list_row_objective.append(list_images[0].list_row_objective[row])

        objective = sum(list_row_objective)

        return ImageData(image, image_ids,
                         self.col_num,
                         objective,
                         list_row_objective,
                         row_shift)

    def generate_image(self):

        image = []
        image_ids = []
        for i in xrange(self.row_num):
            col_num = 0
            is_small_space = 0
            row_image = []
            row_image_ids = []
            while col_num < self.col_num:
                img_id = random.choice(self.char_data.map_img_id_data.keys())

                can_white = check_white(self.image_matrix, self.fontsize, i, col_num, numpy.random.random())
                #can_white = self._check_white(i, col_num)

                if can_white != -1:
                    img_id = can_white
                ret = numpy.empty(2, dtype="i8")
                img_id, is_small_space = check_double_small_space(img_id, is_small_space, ret)
                img = self.char_data.map_img_id_data[img_id]
                row_image.append(img)
                row_image_ids.append(CharCell(img_id,
                                              i,
                                              (col_num, col_num + img.shape[1]),
                                              img.shape[1]
                                             )
                                     )

                col_num = col_num + img.shape[1]
            #while end
            image.append(row_image)
            image_ids.append(row_image_ids)
        # for end

        return ImageData(image, image_ids, self.col_num)

    def _check_white(self, row, col):

        large_white = self.image_matrix[row * self.fontsize:(row+1) * self.fontsize,
                                        col:col+LARGE_SPACE_WIDTH].sum()

        if large_white == 0 and random.random < LARGE_SPACE_RATE:
            return LARGE_SPACE_INDEX

        small_white = self.image_matrix[row * self.fontsize:(row+1) * self.fontsize,
                                        col:col+SMALL_SPACE_WIDTH].sum()

        if small_white == 0:
            return SMALL_SPACE_INDEX

        return -1

    def calc_objective(self, image_data, row_shift=None):

        size = image_data.image_matrix.shape[0]
        min_objective = 1.0e15

        if row_shift is None:
            for i in xrange(self.fontsize):
                compare_matrix = self.image_matrix[i:size+i, :]

                obj = calc_objective(image_data.image_matrix, compare_matrix, WEIGHT)
                if obj < min_objective:
                    min_objective = obj
                    row_shift = i
        else:
                compare_matrix = self.image_matrix[row_shift:size+row_shift, :]
                min_objective = calc_objective(image_data.image_matrix, compare_matrix, WEIGHT)

        list_row_objctive = []
        for row in xrange(self.row_num):
            compare_matrix = self.image_matrix[row * self.fontsize + row_shift:
                                               (row + 1) * self.fontsize + row_shift, :]
            now_image = image_data.image_matrix[row * self.fontsize: (row + 1) * self.fontsize]

            obj = calc_objective(now_image, compare_matrix, WEIGHT)

            list_row_objctive.append(obj)

        return min_objective, list_row_objctive, row_shift

    def local_search(self, image_data, row_shift=0):

        for row, char_cells in enumerate(image_data.image_ids):
            aaa = self._row_local_search(image_data, row, char_cells, row_shift)
            for min_obj, row, col, min_char_id, min_cell, row_start, row_end, col_start, col_end in aaa:
                image_data.image_ids[row][col].char_id = min_char_id
                image_data.image[row][col] = self.char_data.map_img_id_data[min_char_id]
                image_data.image_matrix[row_start-row_shift:row_end-row_shift, col_start:col_end] = min_cell

        return image_data

    def _row_local_search(self, image_data, row, char_cells, row_shift):

        ret_data = []
        for j, char_cell in enumerate(char_cells):

            row_start = char_cell.row * self.fontsize + row_shift
            row_end = row_start + self.fontsize
            col_start = char_cell.char_cell[0]
            col_end = char_cell.char_cell[1]
            width = char_cell.size

            if col_end > self.image_matrix.shape[1]:
                width = width - (col_end - self.image_matrix.shape[1])
                col_end = self.image_matrix.shape[1]

            original_cell = self.image_matrix[row_start:row_end, col_start:col_end]


            min_char_id = char_cell.char_id
            min_cell = self.char_data.map_img_id_data[char_cell.char_id][:, :width]

            min_obj = calc_objective(min_cell, original_cell, WEIGHT)
            is_change = False

            if (row_start, row_end, col_start, col_end) in self._cache_cell_obj:
                min_obj, min_char_id, min_cell = \
                                    self._cache_cell_obj[row_start,
                                                         row_end,
                                                         col_start, col_end]

            else:
                for char_id in self.char_data.map_size_img_id[char_cell.size]:
                    cell = self.char_data.map_img_id_data[char_id][:, :width]

                    obj = calc_objective(cell, original_cell, WEIGHT)

                    if obj < min_obj:
                        is_change=True
                        min_obj = obj
                        min_char_id = char_id
                        min_cell = cell

                self._cache_cell_obj[row_start, row_end, col_start, col_end] = min_obj, min_char_id, min_cell
            if is_change:
                ret_data.append((min_obj, row, j, min_char_id, min_cell, row_start, row_end, col_start, col_end))

        return ret_data

def toapply(cls, mtd_name, *args, **kwargs):
    return getattr(cls, mtd_name)(*args, **kwargs)

class CharCell(object):

    def __init__(self,
                 char_id,
                 row,
                 char_cell,
                 size):

        self.char_id = char_id
        self.row = row
        self.char_cell = char_cell
        self.size = size


class ImageData(object):

    def __init__(self,
                image,
                image_ids,
                col_num,
                objective=None,
                list_row_objective=None,
                row_shift=None
                ):

        self.image = image
        self.image_ids = image_ids
        self.col_num = col_num
        self.image_matrix = self._calc_image_matrix()

        self.objective = objective
        self.list_row_objective = list_row_objective
        self.row_shift = row_shift

    def save_data(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    def make_text(self, word_dic):
        text= ""
        for row in self.image_ids:
            for cell in row:
                text += word_dic[cell.char_id]
            text += "\n"
        return text
        
    def show(self, label=""):
        plt.clf()
        plt.imshow(self.image_matrix, cmap=cm.Greys)
        plt.title(label)
        plt.show()

    def draw(self, label="", fig_name="no1.png"):
        plt.clf()
        plt.imshow(self.image_matrix, cmap=cm.Greys)
        plt.title(label)
        #plt.draw()
        try:
            plt.savefig(fig_name)
        except:
            logger.warn("fail to save fig.")
            pass
    def _calc_image_matrix(self):

        image_matrix = None
        for img in self.image:

            tmp = numpy.concatenate(img, axis=1)
            if image_matrix is None:
                image_matrix = tmp[:, :self.col_num]
            else:
                try:
                    image_matrix = numpy.r_[image_matrix, tmp[:, :self.col_num]]
                except:
                    print self.col_num
                    print image_matrix.shape
                    print tmp[:, :self.col_num].shape
                    raise
        return image_matrix