
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy
import matplotlib.image as mpimg
import glob
import sys


images = glob.glob(sys.argv[1] + "/*.png")
print images

def animate(nframe):
   plt.cla()
   plt.subplots_adjust(left=0., bottom=0., right=1., top=1., wspace = 0., hspace= 0.)
   plt.tick_params(labelbottom='off')
   plt.tick_params(labelleft='off')
   plt.tick_params(length=0)

   img=mpimg.imread(images[nframe])
   plt.imshow(img)

fig = plt.figure(figsize=(12, 8), facecolor="b")  

anim = animation.FuncAnimation(fig, animate, frames=len(images))
anim.save('%s.gif'%sys.argv[1], writer='imagemagick', fps=1);