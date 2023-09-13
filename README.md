### MMOPSO:  Multi-objective particle swarm optimization algorithm using multiple search strategies

##### Reference: Lin Q, Li J, Du Z, et al. A novel multi-objective particle swarm optimization with multiple search strategies[J]. European Journal of Operational Research, 2015, 247(3): 732-744.

| Variables | Meaning                                              |
| --------- | ---------------------------------------------------- |
| npop      | Population size                                      |
| iter      | Iteration number                                     |
| lb        | Lower bound                                          |
| ub        | Upper bound                                          |
| nobj      | The dimension of objective space                     |
| eta_c     | Spread factor distribution index (default = 20)      |
| eta_m     | Perturbance factor distribution index (default = 20) |
| nvar      | The dimension of decision space                      |
| pop       | Population                                           |
| vel       | Velocities                                           |
| objs      | Objectives                                           |
| W         | Weight vectors                                       |
| z         | Ideal points                                         |
| arch      | Archive                                              |
| arch_objs | The objectives of archive                            |
| pf        | Pareto front                                         |

#### Test problem: DTLZ1

$$
\begin{aligned}
	& k = nvar - nobj + 1, \text{ the last $k$ variables is represented as $x_M$} \\
	& g(x_M) = 100 \left[|x_M| + \sum_{x_i \in x_M}(x_i - 0.5)^2 - \cos(20\pi(x_i - 0.5)) \right] \\
	& \min \\
	& f_1(x) = \frac{1}{2}x_1x_2 \cdots x_{M - 1}(1 + g(x_M)) \\
	& f_2(x) = \frac{1}{2}x_1x_2 \cdots (1 - x_{M - 1})(1 + g(x_M)) \\
	& \vdots \\
	& f_{M - 1}(x) = \frac{1}{2}x_1(1 - x_2)(1 + g(x_M)) \\
	& f_M(x) = \frac{1}{2}(1 - x_1)(1 + g(x_M)) \\
	& \text{subject to} \\
	& x_i \in [0, 1], \quad \forall i = 1, \cdots, n
\end{aligned}
$$



#### Example

```python
if __name__ == '__main__':
    main(100, 1000, np.array([0] * 7), np.array([1] * 7))
```

##### Output:

![](https://github.com/Xavier-MaYiMing/MMOPSO/blob/main/Pareto%20front.png)



