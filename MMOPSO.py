#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/8 10:20
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : MMOPSO.py
# @Statement : Multi-objective particle swarm optimization algorithm using multiple search strategies
# @Reference : Lin Q, Li J, Du Z, et al. A novel multi-objective particle swarm optimization with multiple search strategies[J]. European Journal of Operational Research, 2015, 247(3): 732-744.
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


def cal_obj(pop, nobj):
    # 0 <= x <= 1
    g = 100 * (pop.shape[1] - nobj + 1 + np.sum((pop[:, nobj - 1:] - 0.5) ** 2 - np.cos(20 * np.pi * (pop[:, nobj - 1:] - 0.5)), axis=1))
    objs = np.zeros((pop.shape[0], nobj))
    temp_pop = pop[:, : nobj - 1]
    for i in range(nobj):
        f = 0.5 * (1 + g)
        f *= np.prod(temp_pop[:, : temp_pop.shape[1] - i], axis=1)
        if i > 0:
            f *= 1 - temp_pop[:, temp_pop.shape[1] - i]
        objs[:, i] = f
    return objs


def factorial(n):
    # calculate n!
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)


def combination(n, m):
    # choose m elements from an n-length set
    if m == 0 or m == n:
        return 1
    elif m > n:
        return 0
    else:
        return factorial(n) // (factorial(m) * factorial(n - m))


def reference_points(npop, dim):
    # calculate approximately npop uniformly distributed reference points on dim dimensions
    h1 = 0
    while combination(h1 + dim, dim - 1) <= npop:
        h1 += 1
    points = np.array(list(combinations(np.arange(1, h1 + dim), dim - 1))) - np.arange(dim - 1) - 1
    points = (np.concatenate((points, np.zeros((points.shape[0], 1)) + h1), axis=1) - np.concatenate((np.zeros((points.shape[0], 1)), points), axis=1)) / h1
    if h1 < dim:
        h2 = 0
        while combination(h1 + dim - 1, dim - 1) + combination(h2 + dim, dim - 1) <= npop:
            h2 += 1
        if h2 > 0:
            temp_points = np.array(list(combinations(np.arange(1, h2 + dim), dim - 1))) - np.arange(dim - 1) - 1
            temp_points = (np.concatenate((temp_points, np.zeros((temp_points.shape[0], 1)) + h2), axis=1) - np.concatenate((np.zeros((temp_points.shape[0], 1)), temp_points), axis=1)) / h2
            temp_points = temp_points / 2 + 1 / (2 * dim)
            points = np.concatenate((points, temp_points), axis=0)
    points = np.where(points != 0, points, 1e-6)
    return points


def nd_sort(objs):
    # fast non-domination sort
    (npop, nobj) = objs.shape
    n = np.zeros(npop, dtype=int)  # the number of individuals that dominate this individual
    s = []  # the index of individuals that dominated by this individual
    ind = 1
    pfs = {ind: []}  # Pareto fronts
    for i in range(npop):
        s.append([])
        for j in range(npop):
            if i != j:
                less = equal = more = 0
                for k in range(nobj):
                    if objs[i, k] < objs[j, k]:
                        less += 1
                    elif objs[i, k] == objs[j, k]:
                        equal += 1
                    else:
                        more += 1
                if less == 0 and equal != nobj:
                    n[i] += 1
                elif more == 0 and equal != nobj:
                    s[i].append(j)
        if n[i] == 0:
            pfs[ind].append(i)
    return pfs[ind]


def crowding_distance(objs):
    # crowding distance
    (npop, nobj) = objs.shape
    cd = np.zeros(npop)
    fmin = np.min(objs, axis=0)
    fmax = np.max(objs, axis=0)
    df = fmax - fmin
    for i in range(nobj):
        if df[i] != 0:
            rank = np.argsort(objs[:, i])
            cd[rank[0]] = np.inf
            cd[rank[-1]] = np.inf
            for j in range(1, npop - 1):
                cd[rank[j]] += (objs[rank[j + 1], i] - objs[rank[j], i]) / df[i]
    return cd


def update_archive(pop, objs, npop):
    # update the archive
    nondominated = nd_sort(objs)
    if len(nondominated) <= npop:
        return pop[nondominated], objs[nondominated]
    pop = pop[nondominated]
    objs = objs[nondominated]
    cd = crowding_distance(objs)
    rank = np.argsort(-cd)
    return pop[rank[: npop]], objs[rank[: npop]]


def get_best(arch, arch_objs, W, z):
    # get the personal and global best from the archive
    t_obj = arch_objs - z
    norm = np.sqrt(np.sum(W ** 2, axis=1))
    d1 = np.matmul(t_obj, np.transpose(W)) / norm
    d2 = np.sqrt(np.tile(np.sum(t_obj ** 2, axis=1).reshape((arch.shape[0], 1)), (1, W.shape[0])) - d1 ** 2)
    PBI = d1 + 5 * d2
    ind = np.argmin(PBI, axis=0)
    pbest = arch[ind]
    ind = np.random.randint(0, arch.shape[0], W.shape[0])
    gbest = arch[ind]
    return pbest, gbest


def PSO(pop, vel, lb, ub, pbest, gbest, nobj):
    # PSO operator
    [npop, nvar] = pop.shape
    W = np.tile(np.random.uniform(0.1, 0.5, (npop, 1)), (1, nvar))
    r1 = np.tile(np.random.random((npop, 1)), (1, nvar))
    r2 = np.tile(np.random.random((npop, 1)), (1, nvar))
    C1 = np.tile(np.random.uniform(1.5, 2, (npop, 1)), (1, nvar))
    C2 = np.tile(np.random.uniform(1.5, 2, (npop, 1)), (1, nvar))
    off_vel = W * vel
    temp = np.tile(np.random.random((npop, 1)) < 0.7, (1, nvar))
    off_vel[temp] += C1[temp] * r1[temp] * (pbest[temp] - pop[temp])
    off_vel[~temp] += C2[~temp] * r2[~temp] * (gbest[~temp] - pop[~temp])
    off = pop + off_vel
    off = np.min((off, np.tile(ub, (npop, 1))), axis=0)
    off = np.max((off, np.tile(lb, (npop, 1))), axis=0)
    return off, off_vel, cal_obj(off, nobj)


def crossover(mating_pool, lb, ub, eta_c):
    # simulated binary crossover (SBX)
    (noff, dim) = mating_pool.shape
    nm = int(noff / 2) if int(noff / 2) % 2 == 0 else int(noff / 2) - 1
    parent1 = mating_pool[:nm]
    parent2 = mating_pool[nm: 2 * nm]
    beta = np.zeros((nm, dim))
    mu = np.random.random((nm, dim))
    flag1 = mu <= 0.5
    flag2 = ~flag1
    beta[flag1] = (2 * mu[flag1]) ** (1 / (eta_c + 1))
    beta[flag2] = (2 - 2 * mu[flag2]) ** (-1 / (eta_c + 1))
    offspring1 = (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2
    offspring2 = (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2
    offspring = np.concatenate((offspring1, offspring2), axis=0)
    offspring = np.min((offspring, np.tile(ub, (2 * nm, 1))), axis=0)
    offspring = np.max((offspring, np.tile(lb, (2 * nm, 1))), axis=0)
    return offspring


def mutation(pop, lb, ub, eta_m):
    # polynomial mutation
    (npop, dim) = pop.shape
    lb = np.tile(lb, (npop, 1))
    ub = np.tile(ub, (npop, 1))
    site = np.random.random((npop, dim)) < 1 / dim
    mu = np.random.random((npop, dim))
    delta1 = (pop - lb) / (ub - lb)
    delta2 = (ub - pop) / (ub - lb)
    temp = np.logical_and(site, mu <= 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (1 - delta1[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1)
    temp = np.logical_and(site, mu > 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (1 - delta2[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)))
    pop = np.min((pop, ub), axis=0)
    pop = np.max((pop, lb), axis=0)
    return pop


def GA(arch, arch_objs, lb, ub, eta_c, eta_m, nobj):
    # GA operator
    narch = arch.shape[0]
    mating_pool = np.random.permutation(arch[: int(narch / 2)])
    off = crossover(mating_pool, lb, ub, eta_c)
    off = mutation(off, lb, ub, eta_m)
    return np.concatenate((off, arch[int(narch / 2):]), axis=0), np.concatenate((cal_obj(off, nobj), arch_objs[int(narch / 2):]), axis=0)


def main(npop, iter, lb, ub, nobj=3, eta_c=20, eta_m=20):
    """
    The main function
    :param npop: population size
    :param iter: iteration number
    :param lb: lower bound
    :param ub: upper bound
    :param nobj: the dimension of objective space (default = 3)
    :param eta_c: spread factor distribution index (default = 20)
    :param eta_m: perturbance factor distribution index (default = 20)
    :return:
    """
    # Step 1. Initialization
    nvar = len(lb)  # the dimension of decision space
    W = reference_points(npop, nobj)  # weight vectors
    npop = W.shape[0]
    W /= np.sqrt(np.sum(W ** 2, axis=1)).reshape((npop, 1))
    pop = np.random.uniform(lb, ub, (npop, nvar))  # population
    vel = np.zeros((npop, nvar))  # velocities
    objs = cal_obj(pop, nobj)  # objectives
    z = np.min(objs, axis=0)  # ideal point
    arch, arch_objs = update_archive(pop, objs, npop)  # archive

    # Step 2. The main loop
    for t in range(iter):

        if (t + 1) % 200 == 0:
            print('Iteration ' + str(t + 1) + ' completed.')

        # Step 2.1. Get gbest and pbest
        pbest, gbest = get_best(arch, arch_objs, W, z)

        # Step 2.2. PSO
        pop, vel, objs = PSO(pop, vel, lb, ub, pbest, gbest, nobj)
        z = np.min((z, np.min(objs, axis=0)), axis=0)

        # Step 2.3. Update archive
        arch, arch_objs = update_archive(np.concatenate((pop, arch), axis=0), np.concatenate((objs, arch_objs), axis=0), npop)
        temp_arch, temp_arch_objs = GA(arch, arch_objs, lb, ub, eta_c, eta_m, nobj)
        z = np.min((z, np.min(arch_objs, axis=0)), axis=0)
        arch, arch_objs = update_archive(np.concatenate((arch, temp_arch), axis=0), np.concatenate((arch_objs, temp_arch_objs), axis=0), npop)

    # Step 3. Sort the results
    pf = arch_objs[nd_sort(arch_objs)]
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.view_init(45, 45)
    x = [o[0] for o in pf]
    y = [o[1] for o in pf]
    z = [o[2] for o in pf]
    ax.scatter(x, y, z, color='red')
    ax.set_xlabel('objective 1')
    ax.set_ylabel('objective 2')
    ax.set_zlabel('objective 3')
    plt.title('The Pareto front of DTLZ1')
    plt.savefig('Pareto front')
    plt.show()


if __name__ == '__main__':
    main(100, 1000, np.array([0] * 7), np.array([1] * 7))
