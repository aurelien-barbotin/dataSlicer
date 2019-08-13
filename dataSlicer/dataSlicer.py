# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 12:52:03 2018

@author: Aurelien
"""

# -*- coding: utf-8 -*-
"""
Demonstrate a simple data-slicing task: given 3D data (displayed at top), select 
a 2D plane and interpolate data along that plane to generate a slice image 
(displayed at bottom). 


"""

## Add path to library (just for examples; you do not need this)
#import initExample

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import matplotlib.pyplot as plt
import os

from dataSlicer.fitters import Fitter,fitters_list
from dataSlicer.plotting import colorbar_plot, MatplotlibWindow
from dataSlicer.extract_tiff import Scan


from PyQt5.QtWidgets import QFileDialog,QSlider,QSpinBox
from PyQt5.QtCore import Qt
from tifffile import imread
from scipy.ndimage import map_coordinates


colors = ["purple","black","green","orange","brown","pink","gray","olive",
          "cyan","blue","red"]
    
def get_slice(image,x1,y1,x2,y2):
    """gets a slice of image defined by the segment between w1 and x2"""
    num=np.sqrt((y2-y1)**2+(x2-x1)**2)
    x,y = np.linspace(x1,x2,num),np.linspace(y1,y2,num)
    out = map_coordinates(image,np.vstack((x,y)))
    return out

class ExclusiveCheckBox(QtGui.QWidget):
    def __init__(self,nr,names,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.nr = nr
        self.names = names
        self.widgets=[]
        
        for j in range(self.nr):
            cb = QtGui.QCheckbox(names[j])
            cb.setChecked(False)
            self.widgets.append(cb)
        self.widgets[0].setChecked(True)


class SliceableImage(QtGui.QWidget):
    roiChanged = QtCore.Signal(object)
    imageLoaded = QtCore.Signal(np.ndarray)
    index=0
    def __init__(self,nr=0,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.setAcceptDrops(True)
        self.nrotation = 0
        l = QtGui.QGridLayout()
        self.setLayout(l)
        self.win = pg.GraphicsWindow()
        
        self.axis_order = np.array([0,1,2])
        
        w1 = self.win.addLayout(row=0, col=0)
        self.plot = w1.addViewBox(row=1, col=0, lockAspect=True)
        self.plot.disableAutoRange('xy')
        self.imv1 = pg.ImageItem()
        self.plot.addItem(self.imv1)
        """
        self.plot = self.win.addPlot(lockAspect=True)
        self.imv1 = pg.ImageItem(lockAspect=True)
        self.plot.addItem(self.imv1)"""
        #self.imv1.setResizable(False)
        self.psizes = None
        self.loadButton = QtGui.QPushButton("Load")
        self.loadButton.clicked.connect(self.load_prompt)
        self.dataset_name = "example data"
        
        self.datasetLabel = QtGui.QLineEdit(self.dataset_name)
        def update_name():
            self.dataset_name = self.datasetLabel.text()
        self.datasetLabel.editingFinished.connect(update_name)
        
        l.addWidget(self.win, 1, 0,2,2)
        l.addWidget(self.loadButton,0,0)
        l.addWidget(self.datasetLabel,0,1)
        
        self.imageNumber = nr
        
        self.roi = pg.LineSegmentROI([[10, 64], [120,64]], pen='r')
        
        #if self.imageNumber==0:
        self.plot.addItem(self.roi)
        
        x1 = np.linspace(-30, 10, 128)[:, np.newaxis, np.newaxis]
        x2 = np.linspace(-20, 20, 128)[:, np.newaxis, np.newaxis]
        y = np.linspace(-30, 10, 128)[np.newaxis, :, np.newaxis]
        z = np.linspace(-20, 20, 128)[np.newaxis, np.newaxis, :]
        d1 = np.sqrt(x1**2 + y**2 + z**2)
        d2 = 2*np.sqrt(x1[::-1]**2 + y**2 + z**2)
        d3 = 4*np.sqrt(x2**2 + y[:,::-1]**2 + z**2)
        self.data = (np.sin(d1) / d1**2) + (np.sin(d2) / d2**2) + (np.sin(d3) / d3**2)
        self.all_data = self.data.copy()
        self.data = self.data[:,:,0]
        self.slice=np.ones(3)
       
        self.roi.sigRegionChanged.connect(self.update_roi_master)
        ## Display the data
        self.imv1.setImage(self.data)
        self.plot.autoRange()
        
    def update_roi_master(self):
        if self.imageNumber==0:
            self.roiChanged.emit(self.roi)
            self.get_slice()
            
    def get_slice(self):
        #d2 = self.roi.getArrayRegion(self.data, self.imv1.imageItem, axes=(0,1))
        handles_pos = self.roi.getSceneHandlePositions()
        x1,y1 = handles_pos[0][1].x(),handles_pos[0][1].y()
        x2,y2 = handles_pos[1][1].x(),handles_pos[1][1].y()

        d2 = get_slice(self.data,x1,y1,x2,y2)
        d2=self.roi.getArrayRegion(self.data, self.imv1, axes=(0,1))
        self.slice=d2
        
    def get_slices_from_roi(self,data):
        """Normally used only in the master thing"""
        d2=self.roi.getArrayRegion(data, self.imv1, axes=(0,1))
        return d2
        
    def update_roi_slave(self,roi):
        self.roi.handles = roi.handles
        self.roi.setPos(roi.pos())
        self.roi.getLocalHandlePositions()
        self.get_slice()
        
    def load_prompt(self):
        fname,_ = QFileDialog.getOpenFileName(self,"Select Image file", \
                                                  "","Tiff files (*)")
        if fname:
            self.load(fname)
            
    def load(self,name):
        try:
            # self.all_data = np.array(Image.open(name))
            self.all_data = imread(name)
            self.nrotation = 0
            try:
                s = Scan(name)
                psizes = s.stack_info['pixel_sizes']
                self.psizes = psizes
            except:
                psizes = None
            # import exifread
            # f = open(name, 'rb')
            # Return Exif tags
            # tags = exifread.process_file(f)
            
            if np.all( self.all_data+30>=2**15):
                self.all_data[self.all_data<2**15]=2**15
                self.all_data-=2**15  #The tiff format adds 2**15 to every vale
        except:
            print('Format not supported')
            return
        self.imageLoaded.emit(self.all_data)
        if self.all_data.ndim>2:
                self.data = self.all_data[0,:,:]
                self.frameIndex = 0
        else:
            self.data = self.all_data
        print(self.data.shape)
        try:
            self.imv1.setImage(self.data)
            self.dataset_name = os.path.split(name)[-1]
            self.datasetLabel.setText(self.dataset_name)
        except:
            print("Loading image failed")
        
    def set_image(self,image,name):
        self.all_data = np.squeeze(image)
        self.imageLoaded.emit(self.all_data)
        if self.all_data.ndim>2:
                ind=0
                self.data = self.all_data[ind,:,:]
                self.frameIndex = ind
        else:
            self.data = self.all_data
        print(self.data.shape)
        try:
            self.imv1.setImage(self.data)
            self.dataset_name = name
            self.datasetLabel.setText(self.dataset_name)
        except:
            print("Loading image failed")
            
    def rotate(self):
        self.all_data=np.rot90(self.all_data,axes=(-2,-1))
        self.nrotation += 1
        self.set_image(self.all_data,self.dataset_name)
        
    def change_axis(self,index):
        """Changes the point of view of a 3D stack, ie displays xz instead of xy"""
        if self.all_data.ndim<=2:
            return
        if index == self.axis_order[0]:
            return
        remaining_axis = sorted([x for x in self.axis_order if x!=index])   
        new_axis_order = np.array([index, *remaining_axis])
        newpos=[]
        for i in range(3):
            p = np.where(self.axis_order[i]==new_axis_order)[0][0]
            newpos.append(p)
        self.all_data = np.moveaxis(self.all_data,[0,1,2],[1,2,0])

        self.set_image(self.all_data,self.dataset_name)
        self.axis_order = new_axis_order
        
        
    def change_index(self,index):
        print("index:",index,"all data shape:",self.all_data.shape)
        if self.all_data.ndim<=2:
            return
        else:
            self.index = index
            self.data=self.all_data[index,:,:]
            self.imv1.setImage(self.data)
            self.update_roi_master()
            print("index changed")
            
    def dragEnterEvent(self, e):
        e.accept()
    
    def dropEvent(self,e):
        print("drop",e.mimeData().hasUrls)
        for url in e.mimeData().urls():
            url = str(url.toLocalFile())
            self.load(url)
            
class DataSlicer(QtGui.QWidget):
    changeFrame = QtCore.Signal(int)
    def __init__(self,n_images=2,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)
        self.plotWindow = MatplotlibWindow()
        self.plotWindow.resize(200,200)
        self.imageWindows = list()
        
        self.n_images = n_images
        
        self.nrImagesSpinBox = QSpinBox()
        self.nrImagesLabel = QtGui.QLabel("# Images to analyse")
        self.nrImagesSpinBox.setValue(self.n_images)
        self.nrImagesSpinBox.valueChanged.connect(self.make_images_tab)
        self.nrImagesSpinBox.valueChanged.connect(self.rotation_button_connection)
        

        self.images_tab = QtGui.QWidget()
                
        self.make_images_tab()
        self.make_pixel_unit_tab()
        self.make_fitter_tab()
        self.make_img_management_tab()
        self.make_plot_options_tab()
        
        self.imageScrollBar = QtGui.QWidget()
        self.extractImagesButton = QtGui.QPushButton("extract")
        self.extractImagesButton.clicked.connect(self.extract_images)
        
        self.layout.addWidget(self.imageScrollBar,1,0)
        
        self.layout.addWidget(self.plotWindow,2,0,5,5)
        self.layout.addWidget(self.group_pixel_unit,2,5,1,2)
        self.layout.addWidget(self.image_management_tab,2,7,1,1)
        self.layout.addWidget(self.group_fitter,3,5,1,3)
        self.layout.addWidget(self.extractImagesButton,5,5)
        
        self.layout.addWidget(self.nrImagesLabel,5,6)
        self.layout.addWidget(self.nrImagesSpinBox,5,7)
        
    def make_images_tab(self):
        self.images_tab.deleteLater()
        g = QtGui.QGroupBox('Images')
        l = QtGui.QGridLayout()
        
        self.n_images = self.nrImagesSpinBox.value()
        self.imageWindows = []
        
        for i in range(self.n_images):
            self.imageWindows.append(SliceableImage(nr=i))
            self.imageWindows[-1].roiChanged.connect(self.update_rois)
            self.imageWindows[-1].imageLoaded.connect(self.new_image_loaded)
            #self.imageWindows[-1].resize(300,300)
            l.addWidget(self.imageWindows[-1],0,i)
        g.setLayout(l)
        
        self.images_tab = g
        self.layout.addWidget(self.images_tab,0,0,2,8)
        
        
    def make_plot_options_tab(self):
        g = QtGui.QGroupBox('Plot Options')
        l = QtGui.QGridLayout()
    
        self.cmapLineEdit = QtGui.QLineEdit("hot")
        
        self.colorRangeCheckBox = QtGui.QCheckBox("Standardise Color range")
        self.colorRangeCheckBox.setChecked(False)
        
        l.addWidget(QtGui.QLabel("Colormap"),0,0)
        l.addWidget(self.cmapLineEdit,0,1)
        l.addWidget(self.colorRangeCheckBox,0,2)
        
        g.setLayout(l)
        
        self.plot_tab = g
        self.layout.addWidget(self.plot_tab,4,5,1,3)
        
    def extract_images(self):
        same_figure = True
        
        roi = self.imageWindows[0].roi
        
        localpos = roi.getLocalHandlePositions()
        abspos = roi.pos()
        localposx = [int(x[1].x()+abspos.x()) for x in localpos]
        localposy = [int(x[1].y()+abspos.y()) for x in localpos]
        xx,yy = np.linspace(localposx[0],localposx[1],100),np.linspace(localposy[0],localposy[1],100)
            
        plt.ion()
        images=[]
        titles=[]
        
        for imageWindow in self.imageWindows:
            data = np.fliplr(imageWindow.data)
            images.append(np.rot90(data))
            titles.append(imageWindow.dataset_name)
            
        if self.colorRangeCheckBox.isChecked():
            mm = np.min(np.asarray(images))
            mM = np.max(np.asarray(images))
        else:
            mm = None
            mM = None
        sbval=None  #scalebarval
        if self.checkbox2.isChecked():
            sbval = int(self.pixelSizeBox.text())
        cmap = self.cmapLineEdit.text()
        psizes = self.imageWindows[0].psizes
        if psizes is not None and self.imageWindows[0].nrotation%2==0:
            psizes = psizes[::-1]
        if not same_figure:
            fig1 = plt.figure()
            fig,axes = colorbar_plot(images,titles,scalebarval=sbval,fig=fig1,cmap=cmap,
                                     minval=mm,maxval=mM,psizes = psizes)
            axes_plot = plt.subplots(1,self.n_images)
        else:
            fig,axes_all = plt.subplots(self.n_images,2)
            axes_all = axes_all.reshape(self.n_images,2)
            colorbar_plot(images,titles,scalebarval=sbval,cmap=cmap,
                                     minval=mm,maxval=mM,axes = axes_all[:,0],psizes = psizes)
            axes = axes_all[:,0]
            axes_plot = axes_all[:,1]
        for ax in axes:
            xfactor = 1
            if psizes is not None:
                xfactor = psizes[1]/psizes[0]
            ax.plot(xx* xfactor ,yy,"--",color="white")
        
        #Specify z direction
        
        ax_first = axes[0]
        ax_first.annotate('', xy=(-0.1,0.8), xycoords='axes fraction', xytext=(-0.1, 0), 
            arrowprops=dict(facecolor='black'))
        ax_first.annotate("z", xy=(-0.1, 0.9), xycoords="axes fraction",
                  va="center", ha="center",fontsize=15)
        
        self.update_rois(self.imageWindows[0].roi,axes = axes_plot)
        fig.tight_layout()
        plt.ioff()
        
    def update_rois(self,roi,axes = None,*args,**kwargs):

        
        data=[]
        names = []
        unit = "pixels"
        psize = 1
        if self.checkbox2.isChecked():
            psize = float(self.pixelSizeBox.text())
            unit = "nm"
                
        for imageWindow in self.imageWindows:
            #imageWindow.update_roi_slave(roi)
            data.append(self.imageWindows[0].get_slices_from_roi(imageWindow.data))
            names.append(imageWindow.dataset_name)
        
        for j in range(1,len(self.imageWindows)):
            self.imageWindows[j].update_roi_slave(self.imageWindows[0].roi)
        
        if axes is None:
            fig = self.plotWindow.figure
        
            fig.clf()
            ax = fig.add_subplot(1,1,1)
        else:
            ax=axes[0]
        if self.fitButton.isChecked():
            self.fit_data(ax=ax)
                
        else:
            for d in data:
                ax.plot(np.arange(d.size)*psize,d)
            ax.legend(names)
            ax.set_xlabel("Distance ("+unit+")")
            ax.set_ylabel("Counts")
        
            self.plotWindow.plot()

    def fit_data(self,state=None,ax=None,*args,**kwargs):
        """variable state only to avoid bugs when statechanged called"""
        data=[]
        names = []
        fits = []
        
        unit = "pixels"
        psize=1
        if self.checkbox2.isChecked():
            psize = float(self.pixelSizeBox.text())
            unit = "nm"
            
        fittername = str(self.fitterListComboBox.currentText())
        fitter=Fitter(fittername)
        fitter_info = "Fitter Info:\n"
        for imageWindow in self.imageWindows:
            data.append(self.imageWindows[0].get_slices_from_roi(imageWindow.data))
            popt,xh,yh=fitter.fit(np.arange(len(data[-1])), data[-1],normalise=True )
            yh*=np.max(data[-1])
            b = popt[-2]
            if fittername[:3]=='exp':
                fwhm = 2*np.sqrt(np.log(2)/b)
            else:
                fwhm = 2*np.sqrt(b)  #Case of a lorentzian
            unit = "pixels"
            if self.checkbox2.isChecked():
                psize = float(self.pixelSizeBox.text())
                fwhm*=psize
                unit = "nm"
            fitter_info+="FWHM "+imageWindow.dataset_name+" :"+str(fwhm)+unit+"\n"
            
            names.append(imageWindow.dataset_name)
            #names.append("fit "+imageWindow.dataset_name)
            fits.append((xh,yh))
        self.fitterInfoLabel.setText(fitter_info)
        if ax is None:
            fig = self.plotWindow.figure
            fig.clf()
            ax = fig.add_subplot(1,1,1)
        for i,(d,f) in enumerate(zip(data,fits)):
            ax.plot(np.arange(d.size)*psize,d,'o',color=colors[i])
        for i,(d,f) in enumerate(zip(data,fits)):
            ax.plot(f[0]*psize,f[1],color=colors[i])
        ax.legend(names)
        ax.set_xlabel("Distance ("+unit+")")
        ax.set_ylabel("Counts")
        self.plotWindow.plot()
        
    def make_img_management_tab(self):
        g = QtGui.QGroupBox('Image management')
        l = QtGui.QGridLayout()
        
        self.img_axis_checkboxes = []
        axes=["xy","xz","yz"]
        
        def make_cb_connect(index):
            def checkbox_connect():
                
                print("\n\n\n\nindex",index)
                for i in range(len(self.img_axis_checkboxes)):
                    print("i",i)
                    self.img_axis_checkboxes[i].blockSignals(True)
                    if i!=index:
                        print("different")
                        self.img_axis_checkboxes[i].setChecked(False)
                    else:
                        print("equal")
                        self.img_axis_checkboxes[i].setChecked(True)
                        for img in self.imageWindows:
                            img.change_axis(i)
                    self.img_axis_checkboxes[i].blockSignals(False)
            return checkbox_connect
        
        j=0
        
        for ax in axes:
            cb=QtGui.QCheckBox(ax)
            cb.setChecked(False)
            cb.toggled.connect(make_cb_connect(j))
            self.img_axis_checkboxes.append(cb)
            l.addWidget(cb,1,j)
            j+=1
            
        self.img_axis_checkboxes[0].setChecked(True)
        
        self.rotationButton = QtGui.QPushButton("Rotate images")
        self.rotation_button_connection()
        l.addWidget(self.rotationButton,0,0,1,3)
        g.setLayout(l)
        g.setLayout(l)
        self.image_management_tab = g
        
    def change_data_axis(self):
        if self.all_data.ndim<=2:
            return
        self.da
        
    def rotation_button_connection(self):

        try:
            self.rotationButton.disconnect()
            print("rotationButton disconnected")
        except:
            print("Failure rotation disconnection")
        for iw in self.imageWindows:
            self.rotationButton.clicked.connect(iw.rotate)
    
    def make_pixel_unit_tab(self):
        g = QtGui.QGroupBox('Unit')
        l = QtGui.QGridLayout()
        
        self.checkbox1 = QtGui.QCheckBox("Pixel")
        self.checkbox2 = QtGui.QCheckBox("nm")
        
        self.checkbox1.setChecked(True)
        self.checkbox2.toggled.connect(
                lambda : self.checkbox1.setChecked(not self.checkbox2.isChecked()))
        self.checkbox1.toggled.connect(
            lambda : self.checkbox2.setChecked(not self.checkbox1.isChecked() ))
        self.pixelSizeBox = QtGui.QLineEdit("50")
        val = QtGui.QDoubleValidator()
        self.pixelSizeBox.setValidator(val)
        
        l.addWidget(self.checkbox1,0,0)
        l.addWidget(self.checkbox2,0,1)
        l.addWidget(self.pixelSizeBox,1,1)
        
        g.setLayout(l)
    
        self.group_pixel_unit = g
    
    def make_fitter_tab(self):
        g = QtGui.QGroupBox('Fit data')
        l = QtGui.QGridLayout()
        
        self.fitButton = QtGui.QCheckBox("Fit")
        self.fitButton.stateChanged.connect(self.fit_data)
        self.fitterListComboBox = QtGui.QComboBox()
        for fname in fitters_list:
            self.fitterListComboBox.addItem(fname)
            
        self.fitterInfoLabel = QtGui.QLabel("Fitter Info:")
        l.addWidget(self.fitterListComboBox,0,0)
        l.addWidget(self.fitButton,1,0)
        l.addWidget(self.fitterInfoLabel,0,1,2,2)
        
        g.setLayout(l)
        self.group_fitter = g
        
        #self.fitterListComboBox.currentIndexChanged.connect(lambda:self.update_plot())
    def new_image_loaded(self,image):
        #self.imageScrollBar.disconnect()
        self.imageScrollBar.deleteLater()
        self.imageScrollBar = QSlider(Qt.Horizontal)
        self.imageScrollBar.setMaximum(image.shape[0]-1)
        self.imageScrollBar.setMinimum(0)
        self.imageScrollBar.setValue(0)
        
        def change_frame(ind):
            for imageWindow in self.imageWindows:
                imageWindow.change_index(ind)
        self.imageScrollBar.valueChanged.connect(lambda x: change_frame(x))
        self.layout.addWidget(self.imageScrollBar,1,0)
        

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    
    app = QtGui.QApplication([])
    ## Create window with two ImageView widgets
    win = QtGui.QMainWindow()
    win.resize(800,800)
    win.setWindowTitle('Data Slicer')
    cw = DataSlicer()
    win.setCentralWidget(cw)
    win.show()
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
