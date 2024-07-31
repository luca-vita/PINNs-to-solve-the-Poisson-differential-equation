from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import json
import time

set_log_level(LogLevel.ERROR)

# Load evaluation points
with open("Eval_Points/1D_Poisson_eval-points.json", 'r') as file:
  ordered_points = json.load(file)
  ordered_points = np.sort(np.array(ordered_points).flatten())

def fem_1d_poisson_fenics(n, a, b, ua, ub):
    # Create mesh and define function space
    mesh = IntervalMesh(int(n), a, b)
    V = FunctionSpace(mesh, 'CG', 1)

    def boundary_a(x, on_boundary):
        return  on_boundary and np.isclose(x[0], 0) 

    def boundary_b(x, on_boundary):
        return  on_boundary and np.isclose(x[0], b)

    # Define boundary conditions
    u_a = Expression('ua', degree=1, ua=ua)
    u_b = Expression('ub', degree=1, ub=ub)
    bc = [DirichletBC(V, u_a, boundary_a), DirichletBC(V, u_b, boundary_b)]
    
    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Expression('6 * x[0] * exp(-x[0]*x[0]) - 4*(x[0]*x[0]*x[0]) * exp(-x[0]*x[0]) ', degree = 1)
    a = dot(grad(u), grad(v)) * dx
    L = f * v * dx
    
    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)
    
    return u

# Define the problem parameters
num_elements = [64, 128, 256, 512, 1024, 2048, 4096]
a = 0
b = 1
ua = 0
ub = np.exp(-1)  # Boundary condition at x = 1

def u_e(x):
  return x*np.exp(-x*x)

# Solve the problem
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
axs = axs.flatten()

for i, n in enumerate(num_elements):

    print(f'Solving for n={n}')
    solve_time = 0
    eval_time = 0
    for j in range(10): 
        
        t0 = time.time()
        u = fem_1d_poisson_fenics(n, a, b, ua, ub)
        t1 = time.time()
        solve_time += t1 -t0
        t0 = time.time()
        u_values = np.array([u(x) for x in ordered_points])
        t1 = time.time()
        eval_time += t1 -t0

    mean_solve_time = solve_time / 10
    mean_eval_time = eval_time / 10
    print(f'Mean solve time for n={n}: {mean_solve_time}')
    print(f'Mean eval time for n={n}: {mean_eval_time}')

    #calculate the l2 norm error absolute and relative
    l2_norm_error = np.linalg.norm(u_values - u_e(ordered_points), ord=2)
    l2_norm_error_relative = l2_norm_error / np.linalg.norm(u_e(ordered_points), ord=2)
    print(f'L2 norm error for n={n}: {l2_norm_error}')
    print(f'L2 norm error relative for n={n}: {l2_norm_error_relative}')

    #plotting the aproximation and the exact solution
    axs[i].plot(ordered_points, u_values, label=f'u(x) for n={n}')
    axs[i].plot(ordered_points, u_e(ordered_points), label='u_e(x)', linestyle='--')
    axs[i].set_xlabel('x')
    axs[i].set_ylabel('u(x)')
    axs[i].set_title(f'Solution for n={n}')
    axs[i].legend()
    axs[i].grid(True)

# Hide any unused subplots
for j in range(i+1, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.show()




