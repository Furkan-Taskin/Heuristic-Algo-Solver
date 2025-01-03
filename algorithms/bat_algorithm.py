import random
import numpy as np

def bat_algorithm(
    func,
    num_vars,
    bounds,
    pop_size=30,
    max_iter=50,
    freq_min=0.0,
    freq_max=2.0,
    alpha_bat=0.9,
    gamma_bat=0.9
):
    """Minimal Bat Algorithm."""
    # Başlangıç pozisyonları ve hızlar
    bats = []
    velocities = []
    for _ in range(pop_size):
        pos = [random.uniform(bounds[0], bounds[1]) for _ in range(num_vars)]
        vel = [0.0]*num_vars
        bats.append(pos)
        velocities.append(vel)

    # Loudness (A) ve pulse rate (r)
    loudness = [1.0]*pop_size
    pulse_rate = [random.random() for _ in range(pop_size)]

    fitness = [func(*b) for b in bats]
    best_idx = np.argmin(fitness)
    best_bat = bats[best_idx][:]
    best_val = fitness[best_idx]

    best_value_history = []

    for it in range(max_iter):
        freqs = [
            freq_min + (freq_max - freq_min) * random.random()
            for _ in range(pop_size)
        ]

        for i_bat in range(pop_size):
            f_i = freqs[i_bat]

            # Hız güncelleme
            for d in range(num_vars):
                velocities[i_bat][d] += (bats[i_bat][d] - best_bat[d]) * f_i
                # Konum güncelleme
                bats[i_bat][d] += velocities[i_bat][d]

            # Sınır kontrolü
            for d in range(num_vars):
                bats[i_bat][d] = max(bounds[0], min(bats[i_bat][d], bounds[1]))

            # Rastgele dar çevrede arama (Lokal arama)
            if random.random() > pulse_rate[i_bat]:
                epsilon = 0.001
                for d in range(num_vars):
                    bats[i_bat][d] = best_bat[d] + epsilon * (random.uniform(-1, 1))

            new_fit = func(*bats[i_bat])

            # Daha iyiyse ve rastgele gürültü kontrolü sağlanırsa
            if (new_fit < fitness[i_bat]) and (random.random() < loudness[i_bat]):
                fitness[i_bat] = new_fit
                loudness[i_bat] *= alpha_bat
                # Pulse rate güncelleme
                pulse_rate[i_bat] = pulse_rate[i_bat] * (1 - np.exp(-gamma_bat * it))

            # En iyiye güncelle
            if fitness[i_bat] < best_val:
                best_val = fitness[i_bat]
                best_bat = bats[i_bat][:]

        best_value_history.append(best_val)

    return best_bat, best_val, best_value_history
