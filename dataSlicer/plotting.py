#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:39:23 2019

@author: aurelien
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar
from pyqtgraph.Qt import QtCore, QtGui

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

def colorbar_plot(images_list,titles=None,nrows=1,scalebarval=None,fig=None,
                  cmap=plt.cm.gray,minval=None,maxval=None,scalefrees_list=[],
                  units = "nm",axes = None, show_colorbar = True,cbarpos="right",psizes = None):
    """Displays a plot with colorbars to easily compare STED images.
    Parameters:
        images_list: list of images
        titles: optional, list of strings corresponding to the names of the images
        nrows: optional, number of rows over which images are displayed.
        scalebarval: if not None, should eb the pixel size in nm.
        fig: maptlotlib object in which the plot is plotted
        cmap: colormap
        minval: minimum value of the colorscale
        maxval: maximum value of the colorscale
        scalefrees_list: list of indices of images that should not be scaled by
        minval and maxval
        unit: str, name of units to be displayed on the scalebar
        axes: list, contains the axes where data is to be plotted
        show_colorbar: bool, if False does not display the colorbar. Ironic 
            for a colorbar plot
    Returns:
        fig: matplotlib object, handle to the figure
        axes: list of axes in fig
        """
    if titles is not None:
        assert(len(titles)==len(images_list))
    
    if type(images_list)!=list:
        images_list = [images_list]

    nn=len(images_list)
    nn = nn//nrows
    if fig is None:
        if axes is None:
            fig, axes = plt.subplots(ncols=nn,
                                     nrows=nrows,
                                       sharex=False,
                                       sharey=True,
                                       subplot_kw={"adjustable": "box-forced"})
            if type(axes)!=np.ndarray:
                axes=np.array([axes])
            axes = np.ravel(axes)
    else:
        axes=[]
        for i in range(nn):
            axes.append(fig.add_subplot(nrows,nn,i+1))
        
    for j,ax in enumerate(axes):
        nd = images_list[j].ndim
        extent = np.zeros(2*nd)
        for k in range(nd):
            extent[2*(nd-k-1)+1] = images_list[j].shape[k]
            print("extent",extent)
            if psizes is not None:
                extent[2*(nd-k-1)+1]*=psizes[k]/psizes[0]
                print("psizes",psizes)
        try:#valid colormap
            if j in scalefrees_list:
                img0 = ax.imshow(images_list[j], cmap=cmap, extent = extent)
            else:
                img0 = ax.imshow(images_list[j], cmap=cmap,vmin=minval,vmax=maxval, extent = extent)
        except:
            cmap = "hot"
            if j in scalefrees_list:
                img0 = ax.imshow(images_list[j], cmap=cmap, extent = extent)
            else:
                img0 = ax.imshow(images_list[j], cmap=cmap,vmin=minval,vmax=maxval, extent = extent)
        if titles is not None:
            ax.set_title(titles[j])
        ax.axis("off")
        
        if show_colorbar:    
            divider = make_axes_locatable(ax)
            cax = divider.append_axes(cbarpos, size="5%", pad=0.05)
            plt.colorbar(img0, cax=cax)
            cax.yaxis.set_ticks_position(cbarpos)
    if scalebarval:
        scalebar = ScaleBar(scalebarval,
                            units=units,frameon=False,color='white',
                            location='lower right')
        ax.add_artist(scalebar)
    return fig,axes

class MatplotlibWindow(QtGui.QDialog):
    plot_clicked = QtCore.Signal(int)
    back_to_plot = QtCore.Signal()
    image_clicked = QtCore.Signal(np.ndarray)
    
    def __init__(self, parent=None):
        super().__init__(parent)

        # a figure instance to plot on
        self.figure = Figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # set the layout
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.canvas.mpl_connect('button_press_event', self.onclick)
        self.canvas.mpl_connect('pick_event', self.onpick)
        
    def onclick(self,event):
        try:
            if event.dblclick and event.button==1:
                ns = self.figure.axes[0].get_subplotspec().get_gridspec().get_geometry()
                assert(ns[0]==ns[1])
                ns = ns[0]
                
                            
                #gets the number of the subplot that is of interest for us
                n_sub = event.inaxes.rowNum*ns + event.inaxes.colNum
                self.figure.clf()
                self.plot_clicked.emit(n_sub)
            elif event.button==3: 
               self.back_to_plot.emit()
        except:
           pass
       
    def onpick(self,event):
        artist = event.artist
        if isinstance(artist, AxesImage):
            im = artist
            A = im.get_array()
            print('image clicked', A.shape)
            self.image_clicked.emit(A)
            
    def plot(self):
        self.canvas.draw()
