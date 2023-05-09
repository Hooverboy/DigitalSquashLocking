# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 13:11:52 2022

@author: Jonathan
"""
import PySpin as ps
import numpy as np

#%%

# system = ps.System.GetInstance()
# camlist = system.GetCameras()
# Camera = camlist.GetByIndex(0)
# modelname = Camera.TLDevice.DeviceModelName.GetValue()
# modelid =  Camera.TLDevice.DeviceSerialNumber.GetValue()
# print('Made contact with {:s}: ID: {:s}'.format(modelname,modelid))
# Camera.Init()
# print(Camera.ExposureMode.GetCurrentEntry().GetSymbolic())






# #%%
# dir_Camera = dir(Camera)

# dir_system = dir(system)
# nodemap = system.GetTLNodeMap()
# dir_nodemap = dir(nodemap)

# nodemap_interface = system.GetTLNodeMap()

# nodemap_tldevice = Camera.GetTLDeviceNodeMap()

# #%%
# # test = Camera.AcquisitionMode_SingleFrame()

# node_acquisition_mode = ps.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))

# dir_node_aq_mode = dir(node_acquisition_mode)


# #%%
# test = ps.IsAvailable(node_acquisition_mode)
# test2 = Camera.TLDevice.DeviceModelName.GetAccessMode()


# #%% sakseliste
# # system = ps.System.GetInstance()

# Camera.Init()
# print(ps.CValuePtr(Camera.ExposureTime).ToString())
# # print_node_info(Camera.ExposureTime)

# node_acquisition_mode = ps.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
# test = ps.IsAvailable(node_acquisition_mode)
# Camera.DeInit()


#%%

# test4 = Camera.ExposureTime.GetValue()

# system.bitFormat(12)
        # self.exposureAuto('on')
        # self.gainAuto('on')
        # self.acquisitionMode('single')





#%%


class CameraSystem():
    def __init__(self,name='Camera1'):
        
        self.CameraName = name
        
        self.initialize_hardware()


    def initialize_hardware(self):
        self.system = ps.System.GetInstance() #Instantiate system object
        camlist = self.system.GetCameras()
        if camlist.GetSize() > 0:
            self.cam = camlist.GetByIndex(0)
            modelname = self.cam.TLDevice.DeviceModelName.GetValue()
            modelid =  self.cam.TLDevice.DeviceSerialNumber.GetValue()
            print('Made contact with {:s}: ID: {:s}'.format(modelname,modelid))
        else:
            print('{:d} cameras found'.format(camlist.GetSize()))

        #setup the hardware
        
        self.cam.Init()  #initialize camera, then we don't have to use the quickspin API but can use the genAPI (more generic)
        self.cam.TLStream.StreamBufferHandlingMode.SetValue(ps.StreamBufferHandlingMode_NewestOnly)
        self.bitFormat(8)
        self.exposureAuto('on')
        self.gainAuto('on')
        self.acquisitionMode('single')
        self.exposureMode('timed')
        # self.triggerMode('on')
        # self.triggerSource('hardware',risingedge = True)    
    
    def getStreamValue(self):
        self.streamValue = self.cam.TLStream.StreamBufferHandlingMode.GetValue()
        
    def exposure(self,exposuretime = None):
        '''set or gets the exposure time of the camera, in microseconds'''
        if exposuretime is None:
            return self.cam.ExposureTime.GetValue()
        else:
            minexp = self.cam.ExposureTime.GetMin()
            maxexp = self.cam.ExposureTime.GetMax()
            if exposuretime < minexp or exposuretime > maxexp:
                return False
            else:
                self.cam.ExposureTime.SetValue(exposuretime) 
                return True


    def exposureAuto(self,mode = None):
        '''set or get the exposure auto mode'''
        if mode is None:
            return self.cam.ExposureAuto.GetCurrentEntry().GetSymbolic()
        elif mode.lower() == 'on':
            self.cam.ExposureAuto.SetValue(ps.ExposureAuto_Continuous)
        elif mode.lower() == 'off':
            self.cam.ExposureAuto.SetValue(ps.ExposureAuto_Off)
        else:
            print('Failed to set ExposureAuto, did not understand keyword: ',mode)
            return False
        return True


    def exposureMode(self, mode=None):
        '''sets or gets the exposure mode (not auto or manual exposure)'''
        if mode is None:
            return self.cam.ExposureMode.GetCurrentEntry().GetSymbolic()
        elif mode.lower() == 'timed':
            self.cam.ExposureMode.SetValue(ps.ExposureMode_Timed)
        elif mode.lower() == 'triggerwidth':
            self.cam.ExposureMode.SetValue(ps.ExposureMode_TriggerWidth)
        else:
            print('Failed to set Exposuremode, did not understand keyword: ',mode)
            return False
        return True


    def gain(self,gainvalue = None):
        '''set or gets the gain of the ADC in the camera camera, in dB'''
        if gainvalue is None:
            return self.cam.Gain.GetValue()
        else:
            mingain = self.cam.Gain.GetMin()
            maxgain = self.cam.Gain.GetMax()
            if gainvalue < mingain or gainvalue > maxgain:
                return False
            else:
                print(gainvalue)
                self.cam.Gain.SetValue(gainvalue) 
                return True


    def gainAuto(self,mode = None):
        '''set or get the gain auto mode'''
        if mode is None:
            val = self.cam.GainAuto.GetCurrentEntry().GetSymbolic()
            if val in ['Continuous', 'Once']:
                return 'on'
            elif val == 'Off':
                return 'off'
            else:
                return False
        elif mode.lower() == 'on':
            self.cam.GainAuto.SetValue(ps.GainAuto_Continuous)
        elif mode.lower() == 'off':
            self.cam.GainAuto.SetValue(ps.GainAuto_Off)
        else:
            print('Failed to set GainAuto, did not understand keyword: ',mode)
            return False
        return True


    def blacklevel(self,blackvalue = None):
        '''set or gets the blacklevel of camera, in percent'''
        if blackvalue is None:
            return self.cam.BlackLevel.GetValue()
        else:
            minblack = self.cam.BlackLevel.GetMin()
            maxblack = self.cam.BlackLevel.GetMax()
            if blackvalue < minblack or blackvalue > maxblack:
                return False
            else:
                self.cam.BlackLevel.SetValue(blackvalue) 
                return True


    def acquisitionMode(self,mode=None):
        '''sets the acquisitionmode to either 'single','multi', or 'cont'''
        if mode is None:
            return self.cam.AcquisitionMode.GetValue()
        elif mode.lower() == 'single':
            self.cam.AcquisitionMode.SetValue(ps.AcquisitionMode_SingleFrame)
        elif mode.lower() == 'multi':
            self.cam.AcquisitionMode.SetValue(ps.AcquisitionMode_MultiFrame)
        elif mode.lower() == 'cont':
            print('set to cont')
            self.cam.AcquisitionMode.SetValue(ps.AcquisitionMode_Continuous)
        else:
            return False
        return True


    def bitFormat(self, bitsize = None):
        if bitsize is None:
            return (self.cam.PixelFormat.GetCurrentEntry().GetSymbolic(),self.cam.PixelSize.GetCurrentEntry().GetSymbolic())
        elif bitsize == 8:
            print('changing bitformat now to 8')
            self.cam.PixelFormat.SetValue(ps.PixelFormat_Mono8)
            self.bit8 = True
        elif bitsize == 12:
            print('changing bitformat now to 12')
            self.cam.PixelFormat.SetValue(ps.PixelFormat_Mono16)
            self.bit8 = False
        else:
            print('Failed to set Bitformat did not understand keyword: ',bitsize)
            return False
        return True


    def triggerMode(self, mode = None):
        if mode is None:
            return self.cam.TriggerMode.GetCurrentEntry().GetSymbolic()
        elif mode.lower() == 'on':
            self.cam.ExposureMode.SetValue(ps.ExposureMode_Timed)
            self.cam.TriggerMode.SetValue(ps.TriggerMode_On)
        elif mode.lower() == 'off':
            self.cam.TriggerMode.SetValue(ps.TriggerMode_Off)
        else:
            print('Failed to set Triggermode, did not understand keyword: ',mode)
            return False
        return True


    def triggerSource(self,source=None, risingedge = True):
        if source is None:
            return self.cam.TriggerSource.GetCurrentEntry().GetSymbolic()
        elif source.lower() == 'software':
            self.cam.TriggerSource.SetValue(ps.TriggerSource_Software)
        elif source.lower() == 'hardware':
            self.cam.TriggerSource.SetValue(ps.TriggerSource_Line0)
            if risingedge:
                self.cam.TriggerActivation.SetValue(ps.TriggerActivation_RisingEdge)
            else:
                self.cam.TriggerActivation.SetValue(ps.TriggerActivation_FallingEdge)
        else:
            print('Failed to set TriggerSource, did not understand keyword: ',source)
            return False


    def binning(self,bins = None):
        if bins is None:
            return (self.cam.BinningHorizontal.GetValue(),
                    self.cam.BinningVertical.GetValue())
        else:
            horizontalbins = bins[0]
            verticalbins = bins[1]
            self.cam.BinningHorizontal.SetValue(horizontalbins)
            self.cam.BinningVertical.SetValue(verticalbins)
            return True


    def binningMode(self,binningMode = None):
        if binningMode is None:
            return (self.cam.BinningHorizontalMode.GetCurrentEntry().GetSymbolic(),
                    self.cam.BinningVerticalMode.GetCurrentEntry().GetSymbolic())
        elif binningMode.lower() == 'average':
            #self.cam.BinningHorizontalMode.SetValue(ps.BinningHorizontalMode_Average)
            self.cam.BinningVerticalMode.SetValue(ps.BinningVerticalMode_Average)
            return True
        elif binningMode.lower() == 'sum':
            #self.cam.BinningHorizontalMode.SetValue(ps.BinningHorizontalMode_Sum)
            self.cam.BinningVerticalMode.SetValue(ps.BinningVerticalMode_Sum)
            return True
        else:
            print('Failed to set BinningMode, did not understand keyword: ',binningMode)
            return False


    def acquireSingleImage(self):
        if self.cam.AcquisitionMode.GetValue() == ps.AcquisitionMode_SingleFrame:
            self.cam.BeginAcquisition()
            try:
                image_result = self.cam.GetNextImage()
                if image_result.IsIncomplete():
                        print('Image incomplete with image status %d ...' % image_result.GetImageStatus())

                else:
                    if self.bit8:
                        data = np.array(image_result.GetData())
                        print('8 bit')
                    else:
                        newimage = image_result.Convert(ps.PixelFormat_Mono16)
                        #newimage = image_result.Convert(ps.PixelFormat_Mono8)
                        data = np.array(newimage.GetData())
                    image_result.Release()

            except ps.SpinnakerException as ex:
                print('Error: %s' % ex)
                return False

            self.cam.EndAcquisition()
            return data
        else:
            return None


    def beginAcquisition(self):
        if self.cam.AcquisitionMode.GetValue() == ps.AcquisitionMode_Continuous:
            self.cam.BeginAcquisition()


    def endAcquisition(self):
        if self.cam.AcquisitionMode.GetValue() == ps.AcquisitionMode_Continuous:
            self.cam.EndAcquisition()


    def getImage(self):
        if self.cam.AcquisitionMode.GetValue() == ps.AcquisitionMode_Continuous:
            try:
                image_result = self.cam.GetNextImage()
                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
                    return None
                else:
                    if self.bit8:
                        #print(image_result.GetData())
                        data = image_result.GetData()
                        #print(max(data))
                        #print(len(data))
                    else:
                        #print('before:')
                        #print(image_result.GetData())
                        #print(len(image_result.GetData()))
                        #print(max(image_result.GetData()))
                        data = image_result.Convert(ps.PixelFormat_Mono16).GetData()
                        #print('after')
                        #print(max(data))
                        #print(len(data))
                    image_result.Release()       
            except ps.SpinnakerException as ex:
                print('Error: %s' % ex)
                return False
            return data
        else:
            return None
        
    def stop(self):
        self.cam.DeInit()
        del self.cam
        self.system.ReleaseInstance()
        
    
    # def release_instance(self):
    #     self.system.ReleaseInstance()
        
#%%

# CamSys = CameraSystem()
# #%%
# CamSys.exposureAuto('off')
# CamSys.exposureMode('timed')
# CamSys.exposure(10100)

# #%%
# print(CamSys.exposure())

# CamSys.acquisitionMode('single')

# test = CamSys.acquireSingleImage()

# #%%
# CamSys.ReleaseInstance()