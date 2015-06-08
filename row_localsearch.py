# coding: utf-8
import random
import numpy
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict
from numba import jit
import math
SMALL_SPACE_INDEX = 0
SMALL_SPACE_WIDTH = 5
LARGE_SPACE_INDEX = 1
LARGE_SPACE_WIDTH = 5

LARGE_SPACE_RATE = 0.3

    

@jit('f8(i4[:, :], u1[:, :], f8)', nopython=True)
def calc_objective(cell, original_cell, weight):
    
    row = original_cell.shape[0]
    col = original_cell.shape[1]
    ret = 0.
    for i in range(row):
        for j in range(col):
            #ret += (original_cell[i, j] - cell[i, j] * weight)**2
            tmp = original_cell[i, j] - cell[i, j]
            if tmp < 0:
                ret += (weight * tmp)**2
            else:
                ret += tmp**2
    return ret

@jit('i8[:](i8, i8, i8[:])', nopython=True)
def check_double_small_space(img_id, is_small_space, ret):
    if img_id == LARGE_SPACE_INDEX:
        if is_small_space:
            img_id = LARGE_SPACE_INDEX
            is_small_space = 0
        else:
            is_small_space = 1
    else:
        is_small_space = 0

    ret[0] = img_id
    ret[1] = is_small_space
    return ret


@jit('i8(u1[:, :], i8, i8, i8, f8)', nopython=True)
def check_white(image_matrix, fontsize, row, col, ran):
    large_white = image_matrix[row * fontsize:(row+1) * fontsize,
                                    col:col+LARGE_SPACE_WIDTH].sum()

    if large_white == 0 and ran < LARGE_SPACE_RATE:
        return LARGE_SPACE_INDEX

    small_white = image_matrix[row * fontsize:(row+1) * fontsize,
                                    col:col+SMALL_SPACE_WIDTH].sum()

    if small_white == 0:
        return SMALL_SPACE_INDEX

    return -1

"""
def row_local_search(
                      row,
                      char_cells,
                      row_shift,
                      fontsize,
                      image_matrix,
                      char_data,
                      WEIGHT
                      ):
    print map(type,[
                      row,
                      char_cells,
                      row_shift,
                      fontsize,
                      image_matrix,
                      char_data,
                      WEIGHT] )
    ret_data = []
    col_num = len(char_cells)
    #for j, char_cell in enumerate(char_cells):
    for j in range(col_num):
        char_cell = char_cells[j]
        row_start = char_cell.row * fontsize + row_shift
        row_end = row_start + fontsize
        col_start = char_cell.char_cell[0]
        col_end = char_cell.char_cell[1]
        width = char_cell.size

        if col_end > image_matrix.shape[1]:
            width = width - (col_end - image_matrix.shape[1])
            col_end = image_matrix.shape[1]

        original_cell = image_matrix[row_start:row_end, col_start:col_end]


        min_char_id = char_cell.char_id
        min_cell = char_data.map_img_id_data[char_cell.char_id][:, :width]
        error_martix = original_cell - min_cell * WEIGHT

        min_obj = (error_martix**2).sum()
        is_change = False
        for char_id in char_data.map_size_img_id[char_cell.size]:
            cell = char_data.map_img_id_data[char_id][:, :width]
            error_martix = original_cell - cell * WEIGHT
            #error_martix = numpy.where(error_martix < 0, error_martix*WEIGHT, error_martix)
            obj = (error_martix**2).sum()
            if obj < min_obj:
                is_change=True
                min_obj = obj
                min_char_id = char_id
                min_cell = cell

        if is_change:
            ret_data.append((min_obj, row, j, min_char_id, min_cell, row_start, row_end, col_start, col_end))

    return ret_data
"""
if __name__ == "__main__":

    ret = numpy.empty(2, dtype="i8")
    check_double_small_space(0, 1, ret)