from fenics import *
import numpy as np

# Define the potential function
def V(x, y):
    return 0.0 if abs(x) < 0.5 and abs(y) < 0.5 else 1.0

# Define the domain and grid size
xmin, xmax = -1.0, 1.0
ymin, ymax = -1.0, 1.0
Nx, Ny = 100, 100

# Create the finite element mesh
mesh = RectangleMesh(Point(xmin, ymin), Point(xmax, ymax), Nx, Ny)

# Define the finite element space
Vh = FunctionSpace(mesh, "Lagrange", 1)

# Define the trial and test functions
ψ = TrialFunction(Vh)
ϕ = TestFunction(Vh)

# Define the weak form of the 2D Schrödinger equation
a = (inner(grad(ψ), grad(ϕ)) * dx
     - (1.0 + V(x, y)) * ψ * ϕ * dx)
L = 0.0 * ϕ * dx

# Solve the finite element problem
ψh = Function(Vh)
solve(a == L, ψh)

# Plot the solution
ψh_vals = ψh.compute_vertex_values(mesh)
x_coords = mesh.coordinates()[:, 0]
y_coords = mesh.coordinates()[:, 1]
X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
Z = ψh_vals.reshape(X.shape)

import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_surface(X, Y, Z, cmap="viridis")
plt.show()