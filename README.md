# GNSS-INS-FUSION

Implementation and validation of a loosely coupled GNSS/INS integration system using GNSS Single Point Positioning (SPP), INS mechanization, and a 15-state error-state Kalman Filter.

The project processes real UGV datasets including GNSS observations, IMU measurements, and ground truth trajectories.

---

# Project structure

```text
GNSS-INS-FUSION
│
├── data/
│   ├── run2/
│   ├── run3/
│   ├── run4/
│   ├── COD0OPSRAP_*.SP3
│   └── static_imu.txt
│
├── docs/
│
├── output/
│   ├── gnss/
│   ├── imu/
│   ├── kf/
│   └── utils/
│
├── src/
│   ├── gnss/
│   ├── imu/
│   ├── kf/
│   └── utils/
│
├── README.md
└── requirements.txt
```

---

# Installation

Create a Python environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# GNSS processing

## Compute GNSS SPP solution

Run from the project root folder:

```bash
python -m src.gnss.compute_gnss_positions --run 2
```

Available runs:

- `--run 2`
- `--run 3`
- `--run 4`

The script computes the GNSS Single Point Positioning (SPP) solution using pseudorange and Doppler observations.

---

## Generate GNSS plots

```bash
python -m src.gnss.plots --run 2
```


---

# IMU analysis

## Allan deviation analysis

```bash
python -m src.imu.allan_analysis
```

This script computes the Allan deviation of the static IMU dataset and estimates stochastic sensor parameters including:

- Velocity Random Walk (VRW)
- Angle Random Walk (ARW)
- Bias instability

Generated figures are stored in:

```text
output/imu/
```

---

# Kalman Filter integration

## Run the GNSS/INS Kalman Filter

```bash
python -m src.kf.run_kf --run 2
```

Available runs:

- `--run 2`
- `--run 3`
- `--run 4`

The script performs:

1. INS mechanization
2. GNSS/INS integration
3. Error-state Kalman filtering
4. Trajectory and error analysis

---

## Simulate GNSS outages

Example:

```bash
python -m src.kf.run_kf --run 2 --outage-start 120 --outage-duration 60
```

Parameters:

- `--outage-start`: outage start time in seconds
- `--outage-duration`: outage duration in seconds

---

## Measurement covariance sensitivity analysis

Example:

```bash
python -m src.kf.run_kf --run 2 --r-scale 10.0
```

The `--r-scale` parameter multiplies the GNSS measurement covariance matrix to study Kalman filter sensitivity to GNSS uncertainty assumptions.

---

# Outputs

## GNSS outputs

Stored in:

```text
output/gnss/
```



## IMU outputs

Stored in:

```text
output/imu/
```



## Kalman Filter outputs

Stored in:

```text
output/kf/
```


---

# Implemented features

- GNSS Single Point Positioning (SPP)
- Doppler-based velocity estimation
- INS mechanization in ECEF
- Allan deviation IMU characterization
- 15-state error-state Kalman Filter
- GNSS outage simulation
- Q/R covariance tuning analysis
- Ground truth validation

---


GNSS/INS fusion project developed for the DTU Space course: GNSS - 30554