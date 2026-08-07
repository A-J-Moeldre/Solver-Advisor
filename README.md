Solver Advisor
AI‑powered matrix diagnostics and solver recommendations for scientific computing

Solver Advisor is a lightweight analysis tool designed to inspect sparse matrices from scientific simulations (FEM, CFD, optimization, power systems, structural mechanics, etc.) and automatically recommend suitable iterative solvers and preconditioners.

It analyzes matrix structure, symmetry, SPD‑properties, block patterns, condition number, and spectral characteristics using Lanczos iterations.
The tool provides actionable solver recommendations such as CG, GMRES, MINRES, and preconditioners like Jacobi, ILU, SSOR, or Multigrid.

Features
Matrix loading (.mtx, .csv, .npy, .npz)

Symmetry detection

SPD (symmetric positive definite) test

Block‑structure detection

Condition number estimation

Largest/smallest eigenvalue estimation

Lanczos spectrum approximation

Solver recommendation (CG, GMRES, MINRES)

Preconditioner recommendation (Jacobi, SSOR, ILU, Multigrid)

CLI interface

GUI interface (Tkinter)

SuiteSparse matrix downloader (Python script)

Installation
Clone the repository:

Example output:

Kood
Form: (5000, 5000)
Symmetry: True
SPD: True
Block structure: False
Condition number: 1.2e7
Solver: CG
Preconditioner: ILU/Multigrid
GUI
Start the graphical interface:

Kood
python gui/app.py
The GUI allows you to:

Select matrices from the matrices/ folder

Run full diagnostics

View solver recommendations

Inspect eigenvalues, condition number, SPD status, block structure, etc.

Requirements
Python 3.10+, NumPy, SciPy, ssgetpy, Tkinter (included with most Python installations)

Author
Allar‑Joel Möldre  
Numerical Analysis • HPC • Solver Diagnostics

Future Work
AMG preconditioner integration

PETSc backend support

Web‑based GUI

Matrix pattern visualization

Solver performance prediction

Automatic preconditioner tuning
