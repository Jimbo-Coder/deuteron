import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the constants in the problem
hbar = 1.0  # reduced Planck constant
m = 1.0     # mass of the particle
L = 100.0     # length of lattice
V0 = 10  # depth of the potential well

# Define the effective potential
def V_eff(x):
    return(V0 * (np.abs(x) < L/10) + hbar**2 / (2*m*x**2))

# Define the discretization of the spatial domain
N = 100      # number of grid points
dx = L / N   # grid spacing
x = np.linspace(-L/2, L/2, N)  # grid points

# Define the finite-element matrix
A = np.zeros((N, N))
for i in range(N):
    A[i, i] = 2 / (dx**2)
    if i > 0:
        A[i, i-1] = -1 / (dx**2)
    if i < N-1:
        A[i, i+1] = -1 / (dx**2)

# Define the Hamiltonian matrix
H = -hbar**2 / (2*m) * A + np.diag(V_eff(x))

# Calculate the energy eigenvalues and eigenvectors
E, psi = np.linalg.eigh(H)

# Print the energy eigenvalues and eigenvectors
for i in range(1):
    print("Energy eigenvalue:", E[i])
    print("Energy eigenvector:", psi[:, i])

# Define the initial wave function
psi0 = psi[:, 0]  # ground state wave function
#psi0 = psi[:, 0] / np.sqrt(2) + psi[:, 1] / np.sqrt(2) #Scattering State

# Define the time-dependent Schrodinger equation
def dpsi_dt(psi, t):
    return -1j * H.dot(psi)

# Solve the time-dependent Schrodinger equation using the Euler method
dt = 0.01          # time step
num_steps = 100   # number of time steps
psi_t = [psi0]    # list to store the wave functions at each time step
for step in range(num_steps):
    psi_t.append(psi_t[-1] + dt * dpsi_dt(psi_t[-1], step*dt))

# Print the results
# for i in range(3):  # print the first three wave functions
#    print("Wave function at t={:.1f}:".format(i*dt), psi_t[i])

# Create the figure and axes for the plot
fig, ax = plt.subplots()

# Define the function that updates the plot at each time step
l2 = int(np.sqrt(N))
def animate(i):
    ax.clear()
    ax.plot(np.real(psi_t[i].reshape(N,1)))
    ax.set_xlabel('x')
    ax.set_ylabel('Re[$\psi$(x, t)]')

# Create the animation object
ani = animation.FuncAnimation(fig, animate, frames=num_steps+1, interval=100)

# Show the animation
plt.show()

print(A)
print(np.diag(V_eff(x)))

