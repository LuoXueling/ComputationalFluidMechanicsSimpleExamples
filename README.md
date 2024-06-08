# Simple Examples of Computational Fluid Mechanics

This a coursework in 2021 when I was an undergraduate student at Shanghai Jiao Tong University. It implements two 2D finite difference solutions of (i) steady incompressible flow around a circular cylinder (Laplace equation) and (ii) a water tank above a moving plate (Stokes flow using stream function and vorticity equation).

## Steady incompressible flow around a circular cylinder (Laplace equation)

<image src=".assets/cylinder.png" width="50%"></image>

The Laplace equation 

$$
\frac{\partial^2 \Psi}{\partial x^2}+\frac{\partial^2 \Psi}{\partial y^2}=0
$$

is solved using finite difference method (rectangular mesh). 

<image src="Cylinder/0.02.png" width="100%"></image>

(Left bottom: Change of $\Psi$ in each iteration. Left top: Flow field. Right top: velocity. Right mid: u. Right bottom: v)

## A water tank above a moving plate

<image src=".assets/watertank.png" width="50%"></image>

The Stokes flow satisfies

$$
\begin{align*}
&\frac{\partial^2 \xi}{\partial x^2}+\frac{\partial^2 \xi}{\partial y^2}=0 \\
&\frac{\partial^2 \Psi}{\partial x^2}+\frac{\partial^2 \Psi}{\partial y^2}=-\xi \\
&\Psi=0,\frac{\partial \Psi}{\partial y}=U=1 \text{ at the bottom}\\
&\Psi=0,\xi=0 \text{ at the surface}\\
&\Psi=0,\frac{\partial \Psi}{\partial x}=0 \text{ at walls}\\
\end{align*}
$$

The vorticity $\xi$ is solved first and then the stream function $\Psi$ is solved. 

<image src="WaterTank/0.01.png" width="100%"></image>

(Left: Vorticity and change of vorticity in each iteration. Mid: Stream function, velocity vector, and change of stream function in each iteration. Right top: u. Right bottom: v)
