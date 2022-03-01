
import numpy as np
import spectral.io.envi as envi
import numpy.matlib

def Normalization(DarkRefPath, WhiteRefPath, ImPath, saveit=1): 
    DarkRef = envi.open(DarkRefPath)
    WhiteRef = envi.open(WhiteRefPath)
    Im = envi.open(ImPath)
    
    DarkRef = DarkRef.load()
    WhiteRef = WhiteRef.load()
    Im = Im.load()
    
    DM = np.zeros((Im.shape[0],Im.shape[1],Im.shape[2]))
    WM = np.zeros((Im.shape[0],Im.shape[1],Im.shape[2]))
    
    for i in range(Im.shape[2]):
        DM[:,:,i] = np.matlib.repmat(np.mean(DarkRef , 0)[:,i] , Im.shape[0],1)
        WM[:,:,i] = np.matlib.repmat(np.mean(WhiteRef , 0)[:,i] , Im.shape[0],1)
    
    for i in range(Im.shape[2]):
        Im[:,:,i] = (Im[:,:,i] - DM[:,:,i])/(WM[:,:,i]- DM[:,:,i])
    if saveit:
        envi.save_image(os.path.join(os.path.dirname(DarkRefPath) ,'full_data.hdr'), Im)
    return Im
