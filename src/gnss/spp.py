
import src.gnss.utils.rinexReader as rr
import src.gnss.utils.SatOrbits as so

import numpy as np
import pandas as pd
import time
import datetime

CLIGHT = 299792458 # m/s

def _create_kernel(obs, satpos, x):
    """ 
    Create Kernel matrix, A, and
    Data vector, L,
    for NLLS positioning solution
    """
        
    rng1_values = np.linalg.norm(satpos.values - x[:3], axis=1)
    rng1 = pd.DataFrame(rng1_values, index=obs.index)
    
    a1 = (satpos.values - x[:3]) / rng1_values[:, None] 
    a1 = pd.DataFrame(a1, index=satpos.index, columns=['a1x', 'a1y', 'a1z'])
   
    clkErr = pd.Series(np.ones(len(a1)), index=a1.index)
    A = pd.concat([-a1, clkErr.T], axis=1)
    
    L = obs.values.flatten() - rng1.values.flatten() - x[3]
    L = pd.DataFrame(L, index=obs.index, columns=['L']) 
    return L, A


def _spp(obs, satpos, x0):
    """
    Single Point Position solution using Least Squares.
    """
    
    tol = 0.001 
    maxiter = 50 
    
    x = x0
    
    curiter = 0 
    h = np.array([100, 100, 100])

    while np.sum(np.abs(h)) > tol and (curiter < maxiter):

        L, A = _create_kernel(obs, satpos, x)

        AtA = A.values.T @ A.values
        AtL = A.values.T @ L.values.flatten()
        dx = np.linalg.inv(AtA) @ AtL

        x = x+dx
        print(f"Iteration {curiter}: Solution: {x}")
        curiter += 1 

    x = pd.Series(x, index=['X', 'Y', 'Z', 'cdt'], name='Solution') 
    return x


def spp_loop(rinexFile: rr.rinexReader, svpos: so.sp3Orbits, sigTypes: str):

    """Calculate SPP solutions for each epoch in the 
    RINEX file using the satellite positions 
    from the SP3 file.
    """

    x0 = [0, 0, 0, 0] # Guess of x, y, z, AND dt
    sol = {}
    startrun = time.time()

    for epoch in rinexFile.timelist:
        
        obs = rinexFile.get_epoch_data(epoch, oTypes=sigTypes)
        obs = obs.dropna() 
        print(f"Observations for epoch {epoch}: {obs.values.flatten()}")
        tau = obs.loc[:,'C1C'] / CLIGHT
        satpos = svpos.getSvPos(epoch, tau)
        print(f"Satellite positions for epoch {epoch}:\n{satpos}")

        # Split satellite positions and clock errors
        cdts = satpos.iloc[:, 3] * CLIGHT 
        satpos = satpos.iloc[:, :3] 
        
        obs = obs + cdts.values[:, None]
        x = _spp(obs, satpos, x0=x0)
        sol[epoch] = x
    
    endrun = time.time()
    processingtime = round(endrun-startrun, 3)
    print("")
    print(f"Computed {len(rinexFile.timelist)} solutions in: {processingtime} seconds")

    return sol

    
def utc_to_gps_sow(dt):
    """
    Convert UTC to GPS by adding 18 leap seconds.
    """
    gps_epoch = datetime.datetime(1980, 1, 6, 0, 0, 0)
    dt_gpst = dt + datetime.timedelta(seconds=18)
    total_seconds = (dt_gpst - gps_epoch).total_seconds()
    sow = total_seconds % 604800
    return sow




