# README

This folder contains the numeric record for validating the formula of failure probability.

The basic setting is

1. $L=1$, only one resource
2. $r_1=0.5$, $r_2=0.4$, both arms follow Bernoulli Distribution
3. $C=2$

We set $d_1=d_2=d$, we want to know the decreasing trend of failure probability when $d$ converge to 0. 

For deterministic consumption case, we denote $\lfloor\frac{C}{2d}\rfloor$ as the stopping moment and compare $Binomial(\lfloor\frac{C}{2d}\rfloor, r_1)$ and $Binomial(\lfloor\frac{C}{2d}\rfloor, r_2)$ to approximate failure probability.

For uncorrelated stochastic consumption case, we assume the consumption follows Bernoulli Distribution. Then we can use geometry distribution to generate the interval between two consumption. When the consumption achieves C=2, we stops. **Reminds: It's possible that final consumption is caused by the arm 1, for example, the realization of consumptions is the following table**  

| Round Index | 1    | 2    | 3    | 4    |
| ----------- | ---- | ---- | ---- | ---- |
| Arm 1       | 0    | 0    | 0    | 1    |
| Arm 2       | 1    | 0    | 0    | 0    |

**Then we will compare $Binomial(4, r_1)$ and $Binomial(4, r_2)$ as the result of this experiment copy.**

# File Structure

"geometry-sequence": In this folder, the sequence of $d$ is a geometric series. It's suitable to plot the result when the x-coordinate is $\log d$.

"inverse-sequence": In this folder, the sequence of $1/d$ is an arithemetic series. It's suitable to plot the result when the x-coordinate is $-\frac{1}{d}$.

validate-formula-geometric_d.eps and validate-formula-inverse_arithmetic_d.eps: Corresponds to the figure 1 in our paper. You can also find these two figures in the notebooks stored inside  folders "geometry-sequence" and "inverse-sequence"