# Random Walks – Python Package

A Python interface to the high-performance C Random Walk Library

This package provides several random-walk models—including Brownian, correlated, biased, mixed, and time-aware walkers, implemented in optimized C and optionally speeded up via CUDA.
It is designed for movement simulation, ecological modeling, and research workflows (i.e. MoveApps) involving Movebank animal tracking data.

## Installation
### 1. Create & activate a virtual environment
```bash
python -m venv myenv
source myenv/bin/activate
```
## 2. Install the package
```bash
pip install "git+https://github.com/ls11ae/random-walks-python.git#submodules=true"
```
## 3. Check out the Jupyter notebooks ```demos``` or run basic tests
There are Jupyter notebooks (one per random walks model) in the ```demos``` folder that demonstrate the use of the package.

## 4. Write your own code
```python
from random_walk_package.core.BrownianWalker import BrownianWalker

with BrownianWalker(S=7, W=201, H=201, T=100) as walker:
    walker.generate(start_x=100, start_y=100)
    walker.backtrace(end_x=50, end_y=50, plot=True, plot_title="Brownian Walker")
```

## Key Features
- High-performance backend in C with optional CUDA acceleration
- Multiple walker models: Brownian, correlated, biased, mixed, time-aware
- Automatic parameter optimization based on animal type
- Landmarks from MESA landcover classification
- High-resolution weather data from the OpenMeteo project
- Customizable transition matrices per terrain or time step
- Produces interactive leaflet maps for display in your browser
- Efficient serialized data handling for large studies