import jax, flax, optax, time, pickle
import os
import jax.numpy as jnp
from functools import partial
import json
from pyDOE import lhs
from typing import Sequence
from tensorflow_probability.substrates import jax as tfp
import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------------
# Hyperparameters
#----------------------------------------------------
architecture_list = [[5,1],[20,1],[40,1],[5,5,1],[20,20,1],[40,40,1],[5,5,5,1],[20,20,20,1], [40,40,40,1]]
lr = 1e-3
num_epochs = 20000

#----------------------------------------------------
# Define Neural Network Architecture
#----------------------------------------------------
class PINN(flax.linen.Module):
    features: Sequence[int]

    @flax.linen.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = flax.linen.tanh(flax.linen.Dense(feat)(x))
        x = flax.linen.Dense(self.features[-1])(x)
        return x

#----------------------------------------------------
# Define Loss Function
#----------------------------------------------------
# Analytic solution of the 2D Poisson equation
@jax.jit
def u_e(x):
    return x[0]**2 * (x[0]-1)**2 * x[1] * (x[1]-1)**2
# Derivatives for the Neumann Boundary Conditions
@partial(jax.vmap, in_axes=(None, 0, 0,), out_axes=(0, 0, 0))
@jax.jit
def neumann_derivatives(params,xs,ys):
    u = lambda x, y: model.apply(params, jnp.stack((x, y)))
    du_dx_0 = jax.jvp(u,(0.,ys),(1.,0.))[1]
    du_dx_1 = jax.jvp(u,(1.,ys),(1.,0.))[1]
    du_dy_1 = jax.jvp(u,(xs,1.),(0.,1.))[1]
    return du_dx_0, du_dx_1, du_dy_1

# PDE residual for 2D Poisson
@partial(jax.vmap, in_axes=(None, 0, 0), out_axes=0)
@partial(jax.jit, static_argnums=(0,))
def residual(u, x, y):
    H1 = jax.hessian(u, argnums=0)(x,y)
    H2 = jax.hessian(u, argnums=1)(x,y)
    lhs = H1+H2
    rhs = 2*((x**4)*(3*y-2) + (x**3)*(4-6*y) + (x**2)*(6*(y**3)-12*(y**2)+9*y-2) - 6*x*((y-1)**2)*y + ((y-1)**2)*y )
    return lhs - rhs

# Loss functionals
@jax.jit
def pde_residual(params, points):
    return jnp.mean(residual(lambda x, y: model.apply(params, jnp.stack((x, y))), points[:, 0], points[:, 1])**2)


@partial(jax.jit, static_argnums=0)
def pde_true(analytic_sol,params, points):
    return jnp.mean((model.apply(params, jnp.stack((points[:, 0], points[:, 1]), axis=1)) - analytic_sol(points[:, 0], points[:, 1]) )**2)

@jax.jit
def boundary_dirichlet(params, points): # u(x,0) = 0
    return jnp.mean((model.apply(params, jnp.stack((points[:,0],jnp.zeros_like(points[:,1])), axis=1)))**2) 

@partial(jax.jit, static_argnums=0) # du/dx(0,y) = 0, du/dx(1,y) = 0, du/dy(x,1) = 0
def boundary_neumann(neumann_derivatives, params, points):
    du_dx_0, du_dx_1, du_dy_1 = neumann_derivatives(params,points[:,0],points[:,1])
    return jnp.mean((du_dx_0)**2) + jnp.mean((du_dx_1)**2) + jnp.mean((du_dy_1)**2)

#----------------------------------------------------
# Define Training Step
#----------------------------------------------------
@partial(jax.jit, static_argnums=(1, 4))
def training_step(params, opt, opt_state, key, neumann_derivatives):
    lb = jnp.array([0.,0.])
    ub = jnp.array([1.,1.])
    domain_points = lb + (ub-lb)*lhs(2, 256)
    boundary = lb + (ub-lb)*lhs(2, 250)
    
    loss_val, grad = jax.value_and_grad(lambda params: pde_residual(params, domain_points) + 
                                                    boundary_dirichlet(params, boundary) +
                                                    boundary_neumann(neumann_derivatives, params, boundary))(params)
    
    update, opt_state = opt.update(grad, opt_state, params)
    params = optax.apply_updates(params, update)
    return params, opt_state, key


def train_loop(params, adam, opt_state, key, neumann_derivatives):
    for _ in range(num_epochs):
        params, opt_state, key= training_step(params, adam, opt_state, key, neumann_derivatives)
    return params, opt_state, key


#----------------------------------------------------
# Train PINN
#----------------------------------------------------
# Train model 10 times and average over the times

#----------------------------------------------------
with open("./Eval_Points/2D_Poisson_eval-points.json", 'r') as f:
    domain_points = json.load(f)
    domain_points = jnp.array(domain_points)

u_values = jnp.array([u_e(x) for x in domain_points])

fig, axs = plt.subplots(3, 3, figsize=(20, 20),subplot_kw={'projection': '3d'})
axs = axs.flatten()

for i, feature in enumerate(architecture_list):
    tot_eval = 0
    tot_solve = 0
    for _ in range(10):
        #----------------------------------------------------
        # Initialise Model
        #----------------------------------------------------
        model = PINN(feature)
        key, key_ = jax.random.split(jax.random.PRNGKey(17))
        batch_dim = 8
        feature_dim = 2
        params = model.init(key, jnp.ones((batch_dim, feature_dim)))

        #----------------------------------------------------
        # Initialise Optimiser
        #----------------------------------------------------
        adam = optax.adam(lr)
        opt_state = adam.init(params)

        #----------------------------------------------------
        # Start Training with Adam optimiser
        #----------------------------------------------------
        start_time = time.time() 
        tuned_params, opt_state, key = jax.block_until_ready(train_loop(params, adam, opt_state, key, neumann_derivatives))  
        tot_solve = time.time()-start_time

        start_time = time.time()
        u_pred = model.apply(tuned_params, jnp.stack((domain_points[:, 0], domain_points[:, 1]), axis=1)).squeeze()
        tot_eval += time.time()-start_time

    mean_solve_time = tot_solve / 10
    mean_eval_time = tot_eval/ 10
    print(f'Mean solve time for arch={feature}: {mean_solve_time}')
    print(f'Mean eval time for arch={feature}: {mean_eval_time}')

    #calculate the l2 norm error absolute and relative
    l2_norm_error = jnp.linalg.norm(u_pred - u_values, ord=2)
    l2_norm_error_relative = l2_norm_error / jnp.linalg.norm(u_values, ord=2)
    print(f'L2 norm error for arch={feature}: {l2_norm_error}')
    print(f'L2 norm error relative for arch={feature}: {l2_norm_error_relative}')

    #plotting the aproximation and the exact solution
    axs[i].plot_trisurf(domain_points[:,0], domain_points[:,1], u_pred, cmap='viridis')
    axs[i].set_xlabel('x')
    axs[i].set_ylabel('u(x)')
    axs[i].set_title(f'Solution for arch={feature}')
    axs[i].grid(True)

# Hide any unused subplots
for j in range(i+1, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.show()

