# 4179-Toutatis-Rotation-Estimation

Open-source reproduction of Takahashi, Busch & Scheeres (2013), *AJ* **146**:95,
estimating the spin state, moments of inertia, and COM–COF offset of the
tumbling asteroid 4179 Toutatis from its 1992–2008 radar-derived orientation
history, using rigid-body rotational dynamics with Earth and Sun tidal torques.

The pipeline runs a batch least-squares fit of the 3-1-3 Euler-angle
observations, compares the converged 15-state to the published Table 2, and
validates the result against the (unfit) angular-velocity data.

## Requirements

Python 3.9+ and the packages in [requirements.txt](requirements.txt):

```bash
pip install -r requirements.txt
```

`astroquery` is only needed for the one-time ephemeris download.

## Usage

```bash
# 1. Download the JPL Horizons ephemerides into data/ (one-time, needs internet)
python download_ephem.py

# 2. Run the full estimation pipeline
python run_pipeline.py
```

`run_pipeline.py` prints the Table 2 comparison and the angular-velocity
validation, and writes its figures and solution to `out/`.

### Optional

```bash
# Show how also fitting omega over-shrinks the formal covariance (not the paper's approach)
python compare_fit_omega.py
```

## Repository layout

| File | Purpose |
| --- | --- |
| `download_ephem.py` | Fetch Toutatis ephemerides (Sun 1-day, Earth 30-min per flyby) from JPL Horizons into `data/`. |
| `dynamics.py` | Rigid-body rotational dynamics, ephemeris interpolation, and tidal-torque model. |
| `data_paper.py` | Published a-priori state, observations, and constants from Takahashi et al. (2013). |
| `filter.py` | Batch least-squares filter. |
| `run_pipeline.py` | Main entry point: fit, Table 2 comparison, validation, and figures. |
| `compare_fit_omega.py` | Demonstration of double-counting the spin vector in the fit. |

## Outputs

Written to `out/` by `run_pipeline.py`:

- `figure7_euler_residuals.pdf` — pre-fit vs. post-fit Euler-angle residuals.
- `figure8_omega_residuals.pdf` — angular-velocity residuals (validation only).
- `solution.npz` — converged state, covariance, and residuals.

The `data/` and `out/` directories are git-ignored; regenerate them with the
two commands above.

## License

See [LICENSE](LICENSE).
