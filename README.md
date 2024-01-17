# Six-beam-optimizer
Constrained optimization of six-beam scan. 

The mathematical framework defined in:
Sathe, Ameya, et al. "A six-beam method to measure turbulence statistics using ground-based wind lidars." Atmospheric Measurement Techniques 8.2 (2015): 729-740.

Provide the constraint as a minimum elevation (min_beta) and a location of a non-homogeneous region or obstacle to avoid in terms of:
- xmin: distance from the lidar, negative id the lidar is NOT within the region to avoid, negative otherwise
- zmax (only ror xmin<0): height of the region to avoid
- zmin (only for xmax>0): minimum heihgt at which profiles are needed

Scan_optimization.png shows a schematic of the constraints.

N_opt is the number of optimization attempts to perform with initial random starting point. 

The plots show the optimal scan and the evolution of the objective function throughout the N_opt optimizations. The objective function should plateau to ensure that the solution is a global minimum.

Cite as:
"Letizia S., Six-beam profiling scan optimizer, Public git-hub repository, 2024"
