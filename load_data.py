# -*- coding: utf-8 -*-
import cv2
import numpy
from matplotlib import pyplot as plt
from matplotlib import cm
MERGIN = 20

def load_image(path="test.jpg", is_edge=True):
    # 画像をグレースケールで取得
    im_gray = cv2.imread(path ,0)
    
    
    if is_edge:
        # cannyアルゴリズムでエッジ抽出
        _im_edge = cv2.Canny(im_gray,64,128)
    else:
        _im_edge = im_gray
        for i in xrange(_im_edge.shape[0]):
            #_im_edge[i] = numpy.where(_im_edge[i] < 58, 1, 0)
            _im_edge[i] = numpy.where(_im_edge[i] < 128, 1, 0)


    im_edge = numpy.zeros(numpy.array(_im_edge.shape)+MERGIN*2,
                          dtype=numpy.uint8)

    im_edge[MERGIN:-MERGIN, MERGIN:-MERGIN] = _im_edge

    for i in xrange(im_edge.shape[0]):
        im_edge[i] = numpy.where(im_edge[i]==0, 0, 1)
        for j in numpy.where(im_edge[i]==1):
            #im_edge[i, j - 3] = 1
            #im_edge[i, j + 3] = 1
            im_edge[i, j - 2] = 1
            im_edge[i, j + 2] = 1
            im_edge[i, j - 1] = 1
            im_edge[i, j + 1] = 1

    plt.imshow(im_edge, cmap=cm.Greys)
    plt.show()
    return im_edge

    """
    plt.imshow(im_edge, cmap=cm.spectral_r)
    plt.show()

    cv2.imwrite('messigray.png',im_edge)
    cv2.imshow("Show Image",im_edge)
    # キー入力待機
    cv2.waitKey(0)
    # ウィンドウ破棄
    cv2.destroyAllWindows()
    """
    
if __name__ == '__main__':
    plt.figure(facecolor="1")
    load_image(path="hamada.jpg", is_edge=False)
