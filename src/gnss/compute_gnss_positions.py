
import src.gnss.utils.rinexReader as rr
import src.gnss.utils.SatOrbits as so
from spp import spp_loop, utc_to_gps_sow

import pandas as pd
from pathlib import Path
import argparse


current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent

argparser = argparse.ArgumentParser(description="Compute SPP solutions from RINEX observations.")
argparser.add_argument("--run", type=int, default=2, help="Run number to process (default: 2)")
args = argparser.parse_args()


consts = ['G'] 
sigTypes = ["C1C"] 
svpos = so.sp3Orbits(f"{project_root}/data/COD0OPSRAP_20261130000_01D_05M_ORB.SP3")

def main():

    RUN_NUMBER = args.run
    filepath = f"{project_root}/data/run{RUN_NUMBER}/run{RUN_NUMBER}.obs"
    rinexFile = rr.rinexReader(filepath)

    rinexFile.readFile(consts, sigTypes)

    print("Computing solutions: ", end="")
    sol = spp_loop(rinexFile, svpos, sigTypes)

    soldf = pd.DataFrame.from_dict(sol, orient='index')
    soldf.index = pd.to_datetime(soldf.index)

    # Convert UTC to GPS
    soldf.index = [utc_to_gps_sow(epoch) for epoch in soldf.index]
    soldf.index.name = "GPSTime"
    soldf = soldf.drop(columns=['cdt'])
    print(soldf.head())

    soldf.to_csv(f"{project_root}/output/gnss/SPP_solutions_run{RUN_NUMBER}.csv")

if __name__ == "__main__":
    main()