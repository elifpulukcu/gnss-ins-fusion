
import src.gnss.utils.rinexReader as rr
import src.gnss.utils.SatOrbits as so
from .doppler_velocity import get_sat_pos_vel, doppler_velocity_solution

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

        dx, *_ = np.linalg.lstsq(A.values, L.values.flatten(), rcond=None)
        h = dx[:3]

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
        
        obs = rinexFile.get_epoch_data(epoch, oTypes=["C1C", "D1C"])
        obs = obs.dropna(subset=["C1C", "D1C"])

        if len(obs) < 4:
            continue

        tau = obs["C1C"] / CLIGHT

        satpos_full = svpos.getSvPos(epoch, tau)

        cdts = satpos_full.iloc[:, 3] * CLIGHT
        satpos = satpos_full.iloc[:, :3]

        common = obs.index.intersection(satpos.index)

        obs = obs.loc[common]
        satpos = satpos.loc[common]

        # Position from SPP
        obs_corr = obs["C1C"] + cdts.loc[common]
        x_pos = _spp(obs_corr, satpos, x0)
        x0 = x_pos[["X", "Y", "Z", "cdt"]].values

        if x_pos is None:
            continue

        receiver_pos = x_pos[["X", "Y", "Z"]].values

        # Satellite velocity
        satpos_v, satvel = get_sat_pos_vel(epoch, tau.loc[common], svpos)

        common_vel = common.intersection(satpos_v.index).intersection(satvel.index)

        if len(common_vel) < 4:
            continue

        vel_sol = doppler_velocity_solution(
            receiver_pos,
            satpos_v.loc[common_vel],
            satvel.loc[common_vel],
            obs.loc[common_vel, "D1C"],
        )

        if vel_sol is None:
            continue

        solution = pd.concat([x_pos, vel_sol])
        sol[epoch] = solution
    
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




