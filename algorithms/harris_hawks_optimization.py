# algorithms/harris_hawks_optimization.py

import random
import numpy as np
import math

def levy_flight(dim):
    beta = 1.5
    sigma = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = 0.01 * np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    step = u / (np.abs(v) ** (1 / beta))
    return step

def harris_hawks_optimization(
    func,
    num_vars,
    bounds,
    pop_size=30,
    max_iter=50,
    e0=1.0
):
    """
    Harris Hawks Optimization (HHO) - Minimal/Gelişmiş sürüm karışımı.
    Döndürür: (best_solution, best_value, best_value_history)
    """

    # bounds -> [lb, ub] veya [[lb1,..],[ub1,..]]
    if not isinstance(bounds[0], list) and not isinstance(bounds[1], list):
        lb = np.array([bounds[0]] * num_vars, dtype=float)
        ub = np.array([bounds[1]] * num_vars, dtype=float)
    else:
        lb = np.array(bounds[0], dtype=float)
        ub = np.array(bounds[1], dtype=float)

    rabbit_location = np.zeros(num_vars)
    rabbit_energy = float("inf")

    # Başlangıç popülasyonu
    X = np.random.uniform(low=0.0, high=1.0, size=(pop_size, num_vars))
    for d in range(num_vars):
        X[:, d] = X[:, d] * (ub[d] - lb[d]) + lb[d]

    best_value_history = []

    # Ana döngü
    for t in range(max_iter):
        # 1) Mevcut popülasyonu değerlendir
        for i in range(pop_size):
            X[i] = np.clip(X[i], lb, ub)
            fitness = func(*X[i])
            if fitness < rabbit_energy:
                rabbit_energy = fitness
                rabbit_location = X[i].copy()

        # 2) Kaçış enerjisi parametresi
        e1 = 2.0 * (1.0 - t / float(max_iter + 1e-9))

        # 3) Yeni konumlar (Exploration / Exploitation)
        for i in range(pop_size):
            e0 = 2 * random.random() - 1
            escaping_energy = e1 * e0

            if abs(escaping_energy) >= 1:
                # Exploration
                q = random.random()
                rand_index = random.randint(0, pop_size - 1)
                x_rand = X[rand_index].copy()

                if q < 0.5:
                    X[i] = x_rand - random.random() * abs(
                        x_rand - 2 * random.random() * X[i]
                    )
                else:
                    x_mean = np.mean(X, axis=0)
                    X[i] = (rabbit_location - x_mean) - random.random() * (
                        (ub - lb) * random.random() + lb
                    )
            else:
                # Exploitation
                r = random.random()
                current_fitness = func(*X[i])

                if r >= 0.5 and abs(escaping_energy) < 0.5:
                    # Hard besiege
                    X[i] = rabbit_location - escaping_energy * abs(
                        rabbit_location - X[i]
                    )

                elif r >= 0.5 and abs(escaping_energy) >= 0.5:
                    # Soft besiege
                    jump_strength = 2 * (1 - random.random())
                    X[i] = (rabbit_location - X[i]) - escaping_energy * abs(
                        jump_strength * rabbit_location - X[i]
                    )

                elif r < 0.5 and abs(escaping_energy) >= 0.5:
                    # Soft besiege + Levy
                    jump_strength = 2 * (1 - random.random())
                    X1 = rabbit_location - escaping_energy * abs(
                        jump_strength * rabbit_location - X[i]
                    )
                    X1 = np.clip(X1, lb, ub)

                    if func(*X1) < current_fitness:
                        X[i] = X1
                    else:
                        X2 = (
                            rabbit_location
                            - escaping_energy
                            * abs(jump_strength * rabbit_location - X[i])
                            + np.multiply(np.random.randn(num_vars), levy_flight(num_vars))
                        )
                        X2 = np.clip(X2, lb, ub)
                        if func(*X2) < current_fitness:
                            X[i] = X2
                else:
                    # Hard besiege + Levy
                    jump_strength = 2 * (1 - random.random())
                    x_mean = np.mean(X, axis=0)
                    X1 = rabbit_location - escaping_energy * abs(
                        jump_strength * rabbit_location - x_mean
                    )
                    X1 = np.clip(X1, lb, ub)

                    if func(*X1) < current_fitness:
                        X[i] = X1
                    else:
                        X2 = (
                            rabbit_location
                            - escaping_energy
                            * abs(jump_strength * rabbit_location - x_mean)
                            + np.multiply(np.random.randn(num_vars), levy_flight(num_vars))
                        )
                        X2 = np.clip(X2, lb, ub)
                        if func(*X2) < current_fitness:
                            X[i] = X2

        best_value_history.append(rabbit_energy)

    best_solution = rabbit_location.tolist()
    best_value = rabbit_energy

    return best_solution, best_value, best_value_history
