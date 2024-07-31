# Solving 1D and 2D Poisson Equations Using Finite Element Method (FEM) and Physics-Informed Neural Networks (PINNs)

## Purpose

This project aims to explore the possibility of using PINNs to solve partial differential equations experimenting with 1D and 2D Poisson equations, a widely spread equation in engineering) and comparing the results with the Finite Element Method (FEM), in terms of accuracy and performance.
## Technologies Used

- **TensorFlow**: For implementing Physics-Informed Neural Networks (PINNs).
- **JIT (Just-In-Time Compilation)**: Optimizes the performance of Python code, especially for numerical computations.
- **Matplotlib**: For plotting and visualizing results.
- **JAX**: For high-performance numerical computing.

## What Was Done

- **1D Poisson Equation**:
  - Implemented using FEM with various mesh sizes.
  - Implemented using PINNs with different neural network architectures.
  - Compared solve times and L2 norm errors.

- **2D Poisson Equation**:
  - Implemented using FEM with various mesh sizes.
  - Implemented using PINNs with different neural network architectures.
  - Compared solve times and L2 norm errors.

## Reference

- **Paper**: The project is based on the methods discussed in this paper: https://arxiv.org/pdf/2302.04107.pdf
## Summary of Results

- **FEM**:
  - Efficient and accurate with lower solve times.
  - L2 norm error decreases with increasing mesh size.

- **PINNs**:
  - Higher solve times compared to FEM, Lower evaluation times for the 2D poisson equation
  - Accuracy improves with more complex architectures, but solve times are significantly longer.


## Conclusion

The FEM method is more reliable in terms of speed and accuracy for solving Poisson equations, while PINNs offer potential, especially when repeated evaluation is needed but require more optimization to match the efficiency of FEM.

For more details, please refer to the [project report](report/Report%20on%20Solving%201D%20and%202D%20Poisson%20Equations%20Using%20Finite%20Element%20Method%20(FEM)%20and%20Physics-Informed%20Neural%20Networks%20(PINNs).pdf).
