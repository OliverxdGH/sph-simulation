# sph-simulation
This repo contains two small SPH codes meant for learning and experimentation:
- **1D shock-tube style SPH** using a cubic spline kernel, Monaghan-style artificial viscosity, and fixed-timestep RK4
- **3D self-gravitating “planet” SPH toy model** with pressure forces, artificial viscosity, spline-softened gravity, RK4, and optional adaptive timestep control

These implementations use an **O(N²)** all-pairs approach and are intended for small particle counts and clarity, not performance.

## Contents

- `sph_1d.py`  
  1D SPH shock-tube style setup. Produces a 2x2 animated GIF (density, energy, pressure, velocity).

- `planet_sph.py`  
  3D SPH toy model that loads particles from a data file (e.g. `Planet600.dat`). Can add spin, create a two-body collision setup, and export 2D/3D GIFs.

## Requirements

- Python 3.10+ (3.9+ likely fine)
- `numpy`
- `matplotlib`
- `pillow` (for GIF writing via Matplotlib)

Install:
```bash
pip install numpy matplotlib pillow
