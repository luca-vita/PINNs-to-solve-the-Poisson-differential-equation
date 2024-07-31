from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import json
import time

set_log_level(LogLevel.ERROR)

# Load evaluation points
with open("Eval_Points/2D_Poisson_eval-points.json", 'r') as file:
  ordered_points = json.load(file)
  ordered_points = np.array(ordered_points)

def fem_2d_poisson_fenics(n):
    # Create mesh and define function space
    mesh = UnitSquareMesh(int(n), int(n))
    V = FunctionSpace(mesh, 'CG', 1)

    def boundary_L(x, on_boundary):
        return  on_boundary and np.isclose(x[0], 0) 

    def boundary_R(x, on_boundary):
        return  on_boundary and np.isclose(x[0], 1) 

    def boundary_U(x, on_boundary):
        return  on_boundary and np.isclose(x[1], 1)

    def boundary_D(x, on_boundary):
        return  on_boundary and np.isclose(x[1], 0) 

    # Define boundary condition
    u_D = Expression('0', degree = 1)

    bc = [DirichletBC(V, u_D, boundary_D)]
    
    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Expression('- 2*( pow(x[0],4)*(3*x[1] - 2) + pow(x[0],3)*(4 - 6*x[1]) + pow(x[0],2)*(6*pow(x[1],3) - 12*pow(x[1],2) + 9*x[1] - 2) - 6*x[0]*pow((x[1]-1),2)*x[1] + pow((x[1]-1),2)*x[1])', degree = 1)
    a = dot(grad(u), grad(v)) * dx
    L = f * v * dx
    
    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)
    
    return u

    # Define the problem parameters
num_elements = [100, 200 , 300, 400 , 500, 600, 700, 800, 900, 1000]

#real sol utrue(x, y) = x**2 (x − 1)**2*y(y − 1)**2.
def u_e(x):
  return x[0]**2 * (x[0] - 1)**2 * x[1] * (x[1] - 1)**2

# Solve the problem
fig, axs = plt.subplots(2, 5, figsize=(20, 20),subplot_kw={'projection': '3d'})
axs = axs.flatten()

for i, n in enumerate(num_elements):

    print(f'Solving for n={n}')
    solve_time = 0
    eval_time = 0
    for j in range(10): 
        
        t0 = time.time()
        u = fem_2d_poisson_fenics(n)
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

    u_true = np.array([u_e(x) for x in ordered_points])
    #calculate the l2 norm error absolute and relative
    l2_norm_error = np.linalg.norm(u_values - u_true, ord=2)
    l2_norm_error_relative = l2_norm_error / np.linalg.norm(u_true, ord=2)
    print(f'L2 norm error for n={n}: {l2_norm_error}')
    print(f'L2 norm error relative for n={n}: {l2_norm_error_relative}')

    #plot the surface of the function as contours
    axs[i].plot_trisurf(ordered_points[:,0], ordered_points[:,1], u_values, cmap='viridis')
    axs[i].set_xlabel('x')
    axs[i].set_ylabel('y')
    axs[i].set_zlabel('u(x, y)')
    axs[i].set_title(f'Solution for n={n}')
    axs[i].grid(True)


# Hide any unused subplots
for j in range(i+1, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.show()

#now plot the true function
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(ordered_points[:,0], ordered_points[:,1], u_true, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x, y)')
ax.set_title('True solution')
ax.grid(True)

plt.show()


