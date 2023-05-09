#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 21:20:24 2023

@author: jonathan
"""

#%%

import sys #needed if you want to interact with the program in command line
# import PySpin as ps
import numpy as np
# from BasicCameraV1 import CameraSystem #class that sets up the camera
from time import perf_counter # used to compute update frequency

import pyqtgraph as pg #plotting module for pyqt

from PyQt5 import QtCore, QtGui, QtWidgets #GUI module
from PyQt5.QtCore import Qt

from PyQt5.QtGui import QIntValidator

from PyQt5.QtWidgets import (QApplication, QWidget, QGridLayout, QVBoxLayout, 
QHBoxLayout, QPushButton, QCheckBox, QSlider, QLineEdit, QComboBox)

# import siglent_psu_api as siglent

from scipy.signal import convolve2d, sawtooth
import skimage.filters

#Crappy workaround. Used to suppress problem of numpy importing twice. Once manually, and once through PySpin
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#make sure our camera class can be imported, as long as the module is in same folder as this script
try:
    sys.path.index(os.getcwd())
except ValueError:
    sys.path.append(os.getcwd())

from Sr2CameraSetup import CameraSystem #class that sets up the camera
import redpitaya_scpi as scpi

# import matplotlib.image
from skimage.io import imsave
from datetime import datetime


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs) #Boilerplate stolen from the interwebs
        
        self.demo = False
        
        if not self.demo:
            self.camera = CameraSystem() #Starting camera using class from the setup script
            #Setting up the camera to suit our needs
            self.camera.acquisitionMode('cont')
            self.camera.exposureAuto('off')
            self.camera.gainAuto('off')
            self.expTime = 3000
            self.camera.exposure(self.expTime)
            self.camera.gain(0)
            self.camera.binningMode('average')
            self.horBin = 4
            self.vertBin = 4
            self.horSize = 2048//self.horBin
            self.vertSize = 1536//self.vertBin
            self.camera.binning(bins=[self.horBin,self.vertBin])
            self.camera.beginAcquisition()
        elif self.demo:
            self.expTime = 50
            self.horBin = 2
            self.vertBin = 2
            self.horSize = 2048//self.horBin
            self.vertSize = 1536//self.vertBin
            
            self.base_image = np.zeros((self.horSize,self.vertSize))
            # kernel = np.ones((11,11))
            # self.base_image[200:-200,150:170] = 130
            # self.base_image[200:-200,-170:-150] = 130
            self.base_image[self.horSize//2,self.vertSize//2] = 1000
            
            # test = main.base_image
            # n = 41
            # kernel = np.ones((n,n))/n**2
            # kernel[0,:]=1
            # for i in range(10):
            # self.base_image = convolve2d(self.base_image,kernel,'same')
            self.base_image = skimage.filters.gaussian(self.base_image, sigma=(61, 61), truncate=2)#, channel_axis=2)
            
            self.base_image = self.base_image/np.max(self.base_image)*200
        
        #scanning setup
        self.scanResolution = 1000
        self.elipScanArray = np.zeros(self.scanResolution)
        self.scanIteration = 0
        self.scanAmplitude = 0.7 #Voltage in V. Check laser current driver for mapping to Current
        self.voltageMidpoint = 0.8
        self.scanningNow = False
        self.lockedBool = False
        self.lockWalkBool = False
        self.lockPointElip = 0
        self.lockWalkdownResolution = 100
        self.lockWalkdownIteration = 0
        self.lockWalkdownCurrentArray = np.zeros(self.lockWalkdownResolution)
        self.epsilon = 1
        self.slopeSign = -1
        
        # active stabilization variables
        self.errorSamples = 10
        self.elipErrorArray = np.zeros(self.errorSamples)
        self.elipControlK_d = 0.002
        self.elipControlK_i = self.elipControlK_d/10
        
        #bookkeeping variables
        self.lastCurrent = np.copy(self.voltageMidpoint)
        self.bookkeepingResolution = 100
        self.currentArray = np.ones(self.bookkeepingResolution)*self.voltageMidpoint
        self.elipArray = np.zeros(self.bookkeepingResolution)
        
        #Red Pitaya voltage modulation
        # self.rp = scpi.scpi('192.168.1.61',port=5000)
        self.rp = scpi.scpi('10.209.64.68',port=5000)
        # set voltage to midpoint so modulation is possible in both directions.
        self.rp.tx_txt(f'ANALOG:PIN AOUT2,{self.voltageMidpoint}')
        
        # Live video UI setup
        self.videoFeed = pg.GraphicsLayoutWidget()
        self.video = self.videoFeed.addViewBox()
        self.video.setAspectLocked(True)
        
        self.img = pg.ImageItem(levels =(0,255))
        self.video.addItem(self.img)
        
        self.makeROI()
        self.updateROI()
        self.ROI.sigRegionChangeFinished.connect(self.updateROI)
        
        # Ellipticity scan graph
        self.makeElipPlot()
        # Fix right plot when resizing window
        self.ElipPlot.vb.sigResized.connect(self.updateViews)
        

        # Live plot
        self.makeLivePlot()
        self.ElipLivePlot.vb.sigResized.connect(self.updateViews)
        
        
        # Vertical box containing elip plot, lock position slider and voltage plot
        self.plotBox = QtWidgets.QVBoxLayout()
        self.plotBox.addWidget(self.ElipPlotWidget)
        self.plotBox.addWidget(self.ElipLivePlotWidget)
        
        #buttons and inputs for control
        self.controlPanel = QHBoxLayout()
        self.epsilonButton = QPushButton('Normalize Elipticity')
        self.epsilonButton.clicked.connect(self.normalizeElip)
        self.controlPanel.addWidget(self.epsilonButton)
        self.scanButton = QPushButton('Scan')
        self.scanButton.clicked.connect(self.scanStart)
        self.controlPanel.addWidget(self.scanButton)
        self.lockButton = QPushButton('Lock')
        self.lockButton.setEnabled(False)
        self.lockButton.clicked.connect(self.lockStartStop)
        self.controlPanel.addWidget(self.lockButton)

        self.controlPanel2 = QHBoxLayout()
        self.slopeMenu = QComboBox()
        self.slopeMenu.addItem('Positive Slope')
        self.slopeMenu.addItem('Negative Slope')
        self.slopeMenu.currentIndexChanged.connect(self.slopeDirection)
        self.controlPanel2.addWidget(self.slopeMenu)
        
        
        # overall layout grid
        layout = QGridLayout()
        layout.addWidget(self.videoFeed,0,0,1,1)
        layout.addLayout(self.plotBox,0,1,2,1)
        layout.addLayout(self.controlPanel,1,0,1,1)
        layout.addLayout(self.controlPanel2,2,0,1,1)
        
        window = QtWidgets.QWidget()
        window.setLayout(layout) 
    
        self.setCentralWidget(window)
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update) 
        self.timer.start()
        
        #end of init
        
    def update(self):
        if self.demo:
            self.lastImage = self.base_image + np.random.randint(-10,10,size=(self.horSize,self.vertSize))
        else:
            self.lastImage = self.camera.getImage().reshape((self.horSize,self.vertSize),order='F')
            # And then casting to float to avoid overflow errors later
            self.lastImage = self.lastImage.astype('int')
            
        self.displayImage = self.lastImage
        self.img.setImage(self.displayImage, levels =(0,255))
        
        self.updateElip()
        
        if (self.scanningNow and self.scanIteration<self.scanResolution):
            self.scanStep()
        
        elif (self.scanIteration>=self.scanIteration and self.scanningNow):
            self.scanButton.setEnabled(True)
            self.lockButton.setEnabled(True)
            self.ElipPlotPlot.setData(self.elipScanArray)
            self.scanIteration = 0
            self.scanningNow = False
            self.lastCurrent = self.voltageMidpoint
            self.rp.tx_txt(f'ANALOG:PIN AOUT2,{self.lastCurrent}')
        
        # walkdown to initial lock current
        if (self.lockedBool and self.lockWalkBool):
            self.lockWalkdown()
        
        ## stabilization routine
        elif (self.lockedBool and not self.lockWalkBool):
            self.lockUpdate()
        
        ## bookkeeping            
        self.currentArray = np.roll(self.currentArray,1)
        self.currentArray[0] = np.copy(self.lastCurrent)
        self.elipArray = np.roll(self.elipArray,1)
        self.elipArray[0] = np.copy(self.lastElip)
        
        ## updating plots
        self.ElipLivePlotPlot.setData(self.elipArray)
        self.CurrentLivePlotPlot.setData(self.currentArray)
        
    def makeROI(self):
        self.ROI = pg.ROI([self.horSize//2,self.vertSize//2],[100,100],pen = pg.mkPen(color='r',width=2),aspectLocked=True)
        self.video.addItem(self.ROI)
        self.ROI.addTranslateHandle([0.5,0.5])
        self.ROI.addScaleHandle([1,0], [0.5,0.5])
    
    def updateROI(self):
        self.sizeROI = np.array(self.ROI.size())
        self.sizeROI = np.ceil(self.sizeROI).astype('int')
        
        base_logic_array = np.ones(self.sizeROI)
        triLower = np.tril(base_logic_array)
        triLower_rot = np.rot90(triLower)
        triUpper = np.triu(base_logic_array)
        triUpper_rot = np.rot90(triUpper)

        self.q1 = triLower * triUpper_rot
        self.q2 = triLower_rot * triLower
        self.q3 = triUpper * triLower_rot
        self.q4 = triUpper_rot * triUpper
        
    def updateElip(self):
        self.dataROI = self.ROI.getArrayRegion(self.displayImage, self.img)
        if all(self.sizeROI==np.shape(self.dataROI)):
            self.lastElip = ( (np.mean(self.q1 * self.dataROI)+np.mean(self.q3 * self.dataROI)) -  self.epsilon*(np.mean(self.q2 * self.dataROI)+np.mean(self.q4 * self.dataROI)) ) 

    def posBarMoved(self):
        self.posBarValue = int(round(self.posBar.value(),0))
        
    def scanStart(self):
        self.scanButton.setEnabled(False)
        self.lockButton.setEnabled(False)
        self.scanningNow = True
        self.scanIteration = 0
        self.scanCurrentArrray  = self.voltageMidpoint + sawtooth(np.linspace(np.pi*1,np.pi*4,self.scanResolution),width=0.5)*self.scanAmplitude
        
    def scanStep(self):
        self.elipScanArray[self.scanIteration] = np.copy(self.lastElip)
        self.lastCurrent = self.scanCurrentArrray[self.scanIteration]
        self.rp.tx_txt(f'ANALOG:PIN AOUT2,{self.lastCurrent}')
        self.scanIteration += 1

    def lockStartStop(self):
        #Stop lock
        if self.lockedBool:
            self.lockButton.setText('Lock')
            self.lockedBool = False
            self.lastCurrent = self.voltageMidpoint
            self.rp.tx_txt(f'ANALOG:PIN AOUT2,{self.lastCurrent}')
            
        #Start lock
        elif not self.lockedBool:
            self.lockButton.setText('Unlock')
            self.lockedBool = True
            self.lockWalkBool = True
            self.lockCurrentInitial = np.copy(self.scanCurrentArrray[self.posBarValue])
            self.lockElipTarget = np.copy(self.elipScanArray[self.posBarValue])
            self.lockWalkdownIteration = 0
            self.lockWalkdownCurrentArray = np.linspace(self.lockCurrentInitial+self.scanAmplitude,self.lockCurrentInitial,self.lockWalkdownResolution)
        
    def lockWalkdown(self):
        self.lastCurrent = self.lockWalkdownCurrentArray[self.lockWalkdownIteration]
        self.lockWalkdownIteration += 1
        self.rp.tx_txt(f'ANALOG:PIN AOUT2,{self.lastCurrent}')
        # print(f'Walking down, step {self.lockWalkdownIteration}, current set to {self.lastCurrent:.3f}')
        if (self.lockWalkdownIteration) >= self.lockWalkdownResolution:
            self.lockWalkBool = False
            
    def lockUpdate(self):
        # updating values used in PI control (maybe PID in the future, depending on needs)
        self.elipError = self.lockElipTarget - self.lastElip
        self.elipErrorArray = np.roll(self.elipErrorArray,1)
        self.elipErrorArray[0] = np.copy(self.elipError)
        self.elipErrorIntegrated = np.sum(self.elipErrorArray)
        
        self.lastCurrent += self.slopeSign * (self.elipControlK_d * self.elipError 
                                            + self.elipControlK_i * self.elipErrorIntegrated)
        self.rp.tx_txt(f'ANALOG:PIN AOUT2,{self.lastCurrent}')
        
        # print(f'Updated current to {self.lastCurrent:.3f}')

    def makeLivePlot(self):
        self.ElipLivePlotWidget = pg.PlotWidget(title = 'Live Elipticity and correction voltage')
        self.ElipLivePlot = self.ElipLivePlotWidget.plotItem
        self.ElipLivePlot.getAxis('left').setLabel('Elipticity',color='#ff0000')
        
        self.CurrentLivePlot = pg.ViewBox()
        self.ElipLivePlot.showAxis('right')
        self.ElipLivePlot.scene().addItem(self.CurrentLivePlot)
        self.ElipLivePlot.getAxis('right').linkToView(self.CurrentLivePlot)
        self.CurrentLivePlot.setXLink(self.ElipLivePlot)
        self.ElipLivePlot.getAxis('right').setLabel('Current', color='#0000ff')
        self.ElipLivePlot.setYRange(-3,3)    
        
        
        self.updateViews()
        self.ElipLivePlotPlot = self.ElipLivePlot.plot(self.elipArray,pen=pg.mkPen(color='#ff0000',width=2))
        
        self.CurrentLivePlotPlot = pg.PlotCurveItem(pen='b')
        self.CurrentLivePlot.addItem(self.CurrentLivePlotPlot)
        self.CurrentLivePlot.setYRange(0,2)    
        
    def makeElipPlot(self):
        self.ElipPlotWidget = pg.PlotWidget(title='Locking scan')
        self.ElipPlot = self.ElipPlotWidget.plotItem
        self.ElipPlot.getAxis('left').setLabel('Elipticity',color='#ff0000')
        
        self.VarPlot = pg.ViewBox()
        self.ElipPlot.showAxis('right')
        self.ElipPlot.scene().addItem(self.VarPlot)
        self.ElipPlot.getAxis('right').linkToView(self.VarPlot)
        self.VarPlot.setXLink(self.ElipPlot)
        self.ElipPlot.getAxis('right').setLabel('Moving Variance', color='#0000ff')
        
        # self.updateViews()
        self.ElipPlotPlot = self.ElipPlot.plot(self.elipScanArray,pen='r')
        self.VarPlot.addItem(pg.PlotCurveItem([], pen='b'))
        
        self.posBar = pg.InfiniteLine(movable=True, angle=90, pen=pg.mkPen(color='#00FF00') )
        self.posBar.sigPositionChangeFinished.connect(self.posBarMoved)
        self.ElipPlot.addItem(self.posBar)
    
    ## Handle view resizing 
    def updateViews(self):
        ## view has resized; update auxiliary views to match
        self.VarPlot.setGeometry(self.ElipPlot.vb.sceneBoundingRect())
        self.CurrentLivePlot.setGeometry(self.ElipLivePlot.vb.sceneBoundingRect())
        ## need to re-update linked axes since this was called
        ## incorrectly while views had different shapes.
        ## (probably this should be handled in ViewBox.resizeEvent)
        self.VarPlot.linkedViewChanged(self.ElipPlot.vb, self.VarPlot.XAxis)
        self.CurrentLivePlot.linkedViewChanged(self.ElipLivePlot.vb, self.CurrentLivePlot.XAxis)
    
    def normalizeElip(self):
        self.epsilon = ( (np.mean(self.q1 * self.dataROI)+np.mean(self.q3 * self.dataROI)) 
                       / (np.mean(self.q2 * self.dataROI)+np.mean(self.q4 * self.dataROI)) )
    def slopeDirection(self,index):
        print(index)
        if index==0:
            self.slopeSign = -1
        elif index==1:
            self.slopeSign = 1
            
    def closeEvent(self, evt):
        # stops the timer so the app closes cleanly
        self.timer.stop()
        # stops camera so it can be started again if running the app again
        if not self.demo:
            self.camera.endAcquisition()
            self.camera.stop()
        

#%%
app = QtWidgets.QApplication(sys.argv)
main = MainWindow()
main.show()
sys.exit(app.exec_())


    