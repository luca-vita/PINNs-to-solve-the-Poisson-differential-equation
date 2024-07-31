import jax, flax, optax, time, pickle
import os
import jax.numpy as jnp
from functools import partial
from pyDOE import lhs
from typing import Sequence
import json
from tensorflow_probability.substrates import jax as tfp
import numpy as onp
import matplotlib.pyplot as plt

#----------------------------------------------------
# Hyperparameters
#----------------------------------------------------
architecture_list = [[5,1],[20,1],[40,1],[5,5,1],[20,20,1],[40,40,1],[5,5,5,1],[20,20,20,1], [40,40,40,1]]
lr = 1e-4
num_epochs = 15000

def u_e(x):
    return x*jnp.exp(-x**2)
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
# Hessian-vector product
def hvp(f, primals, tangents):
    return jax.jvp(jax.grad(lambda x: f(x)[0]), primals, tangents)[1]

# PDE residual for 1D Poisson
@partial(jax.vmap, in_axes=(None, 0), out_axes=0)
@partial(jax.jit, static_argnums=(0,))
def residual(u, x):
    v = jax.numpy.ones(x.shape)
    lhs = hvp(u,(x,),(v,))
    rhs = (-6*x + 4*x**3)*jax.numpy.exp(-x**2)
    return lhs - rhs

# Loss functionals
@jax.jit
def pde_residual(params, points):
    return jnp.mean(residual(lambda x: model.apply(params, x), points)**2)

@jax.jit
def boundary_residual0(params, xs):
    return jnp.mean((model.apply(params, jnp.zeros_like(xs)))**2)

@jax.jit
def boundary_residual1(params, xs):
    return jnp.mean((model.apply(params, jnp.ones_like(xs))-jnp.exp(-1.))**2)

#----------------------------------------------------
# Define Training Step
#----------------------------------------------------
@partial(jax.jit, static_argnums=(1,))
def training_step(params, opt, opt_state, key):
    lb = onp.array(0.)
    ub = onp.array(1.)
    domain_xs = lb + (ub-lb)*lhs(1, 256) #latin hybercube sampling
    boundary_xs = lb + (ub-lb)*lhs(1, 2)

    loss_val, grad = jax.value_and_grad(lambda params: pde_residual(params, domain_xs) + 
                                                    boundary_residual0(params, boundary_xs) +
                                                    boundary_residual1(params, boundary_xs))(params)
    update, opt_state = opt.update(grad, opt_state, params)
    params = optax.apply_updates(params, update)
    return params, opt_state, key, loss_val
def train_loop(params, adam, opt_state, key):
    losses = []
    for _ in range(num_epochs):
        params, opt_state, key, loss_val = training_step(params, adam, opt_state, key)
        losses.append(loss_val.item())
    return losses, params, opt_state, key, loss_val

# Train model 10 times and average over the times

fig, axs = plt.subplots(3, 3, figsize=(20, 10))
axs = axs.flatten()

with open("./Eval_Points/1D_Poisson_eval-points.json", 'r') as f:
        domain_points = json.load(f)
        domain_points = jnp.array(domain_points)
ordered_points = jnp.sort(domain_points.flatten()).reshape(-1,1)
u_values = u_e(ordered_points).squeeze()

times_train_temp = []
times_solve = []
times_eval = []

for i, feature in enumerate(architecture_list):
    
    tot_solve = 0
    tot_eval = 0

    for _ in range(10): 

        #----------------------------------------------------
        # Initialise Model
        #----------------------------------------------------
        model = PINN(feature)
        key, key_ = jax.random.split(jax.random.PRNGKey(17))
        batch_dim = 8
        feature_dim = 1
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
        losses, tuned_params, opt_state, key, loss_val = jax.block_until_ready(train_loop(params, adam, opt_state, key))  #this is only available from jax 0.2.27, but I have 0.2.24 installed
        tot_solve += time.time()-start_time


        # Evaluation and comparison to ground truth

        start_time = time.time()
        u_pred = jax.block_until_ready(model.apply(tuned_params, ordered_points).squeeze()) 
        tot_eval += (time.time()-start_time)

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
    axs[i].plot(ordered_points, u_pred, label=f'u(x) for arch={feature}')
    axs[i].plot(ordered_points, u_values, label='u_e(x)', linestyle='--')
    axs[i].set_xlabel('x')
    axs[i].set_ylabel('u(x)')
    axs[i].set_title(f'Solution for arch={feature}')
    axs[i].legend()
    axs[i].grid(True)

# Hide any unused subplots
for j in range(i+1, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.show()