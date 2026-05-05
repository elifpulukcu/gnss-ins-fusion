
import src.gnss.utils.rinexReader as rr
import src.gnss.utils.SatOrbits as so
from src.gnss.spp import spp_loop, utc_to_gps_sow

import pandas as pd
from pathlib import Path
import argparse


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent

argparser = argparse.ArgumentParser(description="Compute SPP solutions from RINEX observations.")
argparser.add_argument("--run", type=int, default=2, help="Run number to process (default: 2)")
args = argparser.parse_args()


CONSTS = ['G'] 
SIGTYPES = ["C1C"] 
SVPOS = so.sp3Orbits(f"{PROJECT_ROOT}/data/COD0OPSRAP_20261130000_01D_05M_ORB.SP3")

def main():

    RUN_NUMBER = args.run
    filepath = f"{PROJECT_ROOT}/data/run{RUN_NUMBER}/run{RUN_NUMBER}.obs"
    rinexFile = rr.rinexReader(filepath)

    rinexFile.readFile(CONSTS, SIGTYPES)

    print("Computing solutions: ", end="")
    sol = spp_loop(rinexFile, SVPOS, SIGTYPES)

    soldf = pd.DataFrame.from_dict(sol, orient='index')
    soldf.index = pd.to_datetime(soldf.index)

    # Convert UTC to GPS
    soldf.index = [utc_to_gps_sow(epoch) for epoch in soldf.index]
    soldf.index.name = "GPSTime"
    soldf = soldf.drop(columns=['cdt'])
    print(soldf.head())

    soldf.to_csv(f"{PROJECT_ROOT}/output/gnss/SPP_solutions_run{RUN_NUMBER}.csv")

if __name__ == "__main__":
    main()