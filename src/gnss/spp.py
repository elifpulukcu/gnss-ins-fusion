
import src.gnss.utils.rinexReader as rr
import src.gnss.utils.SatOrbits as so
import src.gnss.utils.frame2kml as fk

import numpy as np
import pandas as pd
import time
import datetime
from pathlib import Path

RUN_NUMBER = 2

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent

filepath = f"{project_root}/data/run{RUN_NUMBER}/run{RUN_NUMBER}.obs"
rinexFile = rr.rinexReader(filepath)
svpos = so.sp3Orbits(f"{project_root}/data/COD0OPSRAP_20261130000_01D_05M_ORB.SP3")

clight = 299792458 # m/s

def create_kernel(obs, satpos, x):
    """ 
    Create Kernel matrix, A, and
    Data vector, L,
    for NLLS positioning solution
    
    Parameters
    ----------
    obs : TYPE
        DESCRIPTION.
    satpos : Pandas dataframe
        Contains satellite positions of all satellites currently in view.
    x : numpy array, size [4,]
        Contains position and clock of current iteration of the NLLS solution.

    Returns
    -------
    L : TYPE
        Data vector.
    A : TYPE
        Kernel matrix.

    """
        
    # Determine ranges to all satellites
    rng1_values = np.linalg.norm(satpos.values - x[:3], axis=1)
    rng1 = pd.DataFrame(rng1_values, index=obs.index)
    # Determine unit vectors towards the satellites 
    # (should be a 2D array containing all the unit vectors in the kernel)
    a1 = (satpos.values - x[:3]) / rng1_values[:, None] # Get vector from receiver to satellite
    a1 = pd.DataFrame(a1, index=satpos.index, columns=['a1x', 'a1y', 'a1z']) # Convert to dataframe for easier handling
    # Create a clock error vector and create the full kernel, A
    clkErr = pd.Series(np.ones(len(a1)), index=a1.index)
    A = pd.concat([-a1, clkErr.T], axis=1)
    
    print(f"Observations: {obs.values}")
    print(f"Ranges: {rng1.values}")
    print(f"Current clock error: {x[3]}")
    # Determine the data vector: Obs - ranges - current clock error
    L = obs.values.flatten() - rng1.values.flatten() - x[3] # Account for current clock error in data vector
    print(f"Data vector: {L}")
    L = pd.DataFrame(L, index=obs.index, columns=['L']) # Convert to dataframe for easier handling
    return L, A

def spp(obs, satpos, x0):
    """
    

    Parameters
    ----------
    obs : TYPE
        DESCRIPTION.
    satpos : TYPE
        DESCRIPTION.
    x0 : TYPE
        DESCRIPTION.

    Returns
    -------
    x : TYPE
        DESCRIPTION.

    """
    
    ##### Non-Linear Least Squares SPP #####
    tol = 0.001 # Solution must change less than "tol" in 3D to end the loop
    maxiter = 50 # Maximum number of iterations (10 should be enough)
    
    x = x0
    
    # Initialize the NLLS solution
    curiter = 0 # count number of iterations
    h = np.array([100, 100, 100]) # Select inital h far above tolerance

    while np.sum(np.abs(h)) > tol and (curiter < maxiter):

        # Setup the LS system
        L, A = create_kernel(obs, satpos, x)

        AtA = A.values.T @ A.values
        AtL = A.values.T @ L.values.flatten()
        dx = np.linalg.inv(AtA) @ AtL
        
        # Update solution
        x = x+dx
        print(f"Iteration {curiter}: Solution: {x}")
        curiter += 1 # Keep track of number of iterations

    x = pd.Series(x, index=['X', 'Y', 'Z', 'cdt'], name='Solution') # Setup output
    
    return x


def utc_to_gps_sow(dt):
    gps_epoch = datetime.datetime(1980, 1, 6, 0, 0, 0)
    # Convert UTC to GPST by adding 18 leap seconds
    dt_gpst = dt + datetime.timedelta(seconds=18)
    total_seconds = (dt_gpst - gps_epoch).total_seconds()
    sow = total_seconds % 604800
    return sow

# Select constellations and observations to read
consts = ['G'] # Load GPS data
sigTypes = ["C1C"] # Load only code observations

# Now read the selected contents from the file
rinexFile.readFile(consts, sigTypes)

# Initial guess for position
x0 = [0, 0, 0, 0] # Guess of x, y, z, AND dt

# Initialize a dictionary for storing all solutions
sol = {}

startrun = time.time()

print("Computing solutions: ", end="")

for epoch in rinexFile.timelist:
    
    print(".", end="")

    # Load data from current epoch
    obs = rinexFile.get_epoch_data(epoch, oTypes=sigTypes)
    obs = obs.dropna() # Remove potential nans
    print(f"Observations for epoch {epoch}: {obs.values.flatten()}")
    # Get signal travel time
    tau = obs.loc[:,'C1C'] / clight
    
    # Get satellite positions (only for satellites with observations)
    satpos = svpos.getSvPos(epoch, tau)
    print(f"Satellite positions for epoch {epoch}:\n{satpos}")
    # Split satellite positions and clock errors
    cdts = satpos.iloc[:, 3] * clight # Get satellite clock errors
    satpos = satpos.iloc[:, :3] # Satellite positions (no clocks)
    
    # Account for satellite clock errors
    obs = obs + cdts.values[:, None]
    
    # Get single point positioning
    x = spp(obs, satpos, x0=x0)
    
    # Store SPP in a dictionary
    sol[epoch] = x
    
endrun = time.time()
processingtime = round(endrun-startrun, 3)
print("")
print(f"Computed {len(rinexFile.timelist)} solutions in: {processingtime} seconds")

soldf = pd.DataFrame.from_dict(sol, orient='index')
soldf.index = pd.to_datetime(soldf.index)

# Convert UTC to GPS
soldf.index = [utc_to_gps_sow(epoch) for epoch in soldf.index]
soldf.index.name = "GPSTime"
soldf = soldf.drop(columns=['cdt'])
print(soldf.head())
soldf.to_csv(f"{project_root}/output/gnss/SPP_solutions_run{RUN_NUMBER}.csv")


