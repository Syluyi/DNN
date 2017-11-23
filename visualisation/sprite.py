# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import math
import os


SPRITE_WIDTH = 32
SPRITE_HEIGHT = SPRITE_WIDTH
DPI = 50 #100

def create_sprite(plotables, sprite_file):
    print("creating sprite image")    
    fig= plt.figure(figsize = (SPRITE_WIDTH, SPRITE_HEIGHT), dpi = DPI, frameon=False)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plots_per_side = math.ceil(math.sqrt(len(plotables)))
    plot_width = int(SPRITE_WIDTH * DPI / plots_per_side)
    plot_highth = int(SPRITE_HEIGHT * DPI / plots_per_side)
    for idx, plotable in enumerate(plotables):
        fig.add_subplot(plots_per_side, plots_per_side, idx+1, frame_on=False)
        spectro = plt.imshow(np.transpose(plotable), cmap="jet", vmin=1, vmax=1000, norm=col.LogNorm(), origin="lower", aspect="auto")
        spectro.axes.get_xaxis().set_visible(False)
        spectro.axes.get_yaxis().set_visible(False)
    os.makedirs(os.path.dirname(sprite_file), exist_ok=True)
    plt.savefig(sprite_file, format='png')    
    return  plot_width, plot_highth
