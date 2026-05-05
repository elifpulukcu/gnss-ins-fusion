# gnss-ins-fusion

## Running the GNSS scripts

Run all commands from the project root folder:

```bash
python3 -m src.gnss.compute_gnss_positions --run 4
python3 -m src.gnss.plots
```

The first command computes the SPP solution for a selected run. Use `--run 2`, `--run 3`, or `--run 4` depending on which dataset you want to process.


The second command generates the GNSS trajectory plots. Use `--run 2`, `--run 3`, or `--run 4` depending on which solution you want to plot. If no parameter is specified the plots are generetated for all the runs.

## Outputs

- SPP solutions are written to `output/gnss/SPP_solutions_run<run>.csv`
- Trajectory plots are written to `output/gnss/trajectory_map_run<run>.png`