import random
import numpy as np

def pso(
    func,
    num_vars,
    bounds,
    pop_size=30,
    max_iter=50,
    w_max=0.9,
    w_min=0.2,
    c1=2.0,
    c2=2.0,
    v_max=6.0
):
    """
    Minimal PSO (Parçacık Sürü Optimizasyonu):
      - w (inertial weight) lineer olarak w_max'tan w_min'e iner
      - c1, c2, v_max parametreleri
      - Tekli veya çoklu bounds yönetimi
      - En iyi çözümü (best_solution), en iyi değeri (best_value) ve
        iterasyonlardaki en iyi değer geçmişini (best_value_history) döndürür.
    """

    # bounds tek sayı ise -> her değişken için aynı lb ve ub
    if not isinstance(bounds[0], list) and not isinstance(bounds[1], list):
        lb = np.array([bounds[0]] * num_vars, dtype=float)
        ub = np.array([bounds[1]] * num_vars, dtype=float)
    else:
        lb = np.array(bounds[0], dtype=float)
        ub = np.array(bounds[1], dtype=float)

    # Parçacıkların pozisyonu (pos) ve hızları (vel)
    pos = np.random.uniform(low=lb, high=ub, size=(pop_size, num_vars))
    vel = np.zeros((pop_size, num_vars))

    # Parçacıkların en iyi skorları ve konumları
    p_best_score = np.full(pop_size, np.inf)
    p_best_pos = np.copy(pos)

    # Sürünün genel en iyi skoru ve konumu
    g_best_score = float("inf")
    g_best_pos = np.zeros(num_vars)

    # Iterasyon boyunca en iyi değeri kaydetmek için
    best_value_history = []

    # PSO döngüsü
    for iteration in range(max_iter):
        # Atanacak w değeri (lineer azalma)
        w = w_max - (w_max - w_min) * (iteration / float(max_iter))

        for i in range(pop_size):
            # Sınır kontrolü
            pos[i] = np.clip(pos[i], lb, ub)

            # Her parçacık için fitness hesabı
            fitness = func(*pos[i])

            # Kişisel en iyi güncellemesi
            if fitness < p_best_score[i]:
                p_best_score[i] = fitness
                p_best_pos[i] = pos[i].copy()

            # Küresel en iyi güncellemesi
            if fitness < g_best_score:
                g_best_score = fitness
                g_best_pos = pos[i].copy()

        # Her parçacığın hız ve konum güncellemesi
        for i in range(pop_size):
            for d in range(num_vars):
                r1 = random.random()
                r2 = random.random()
                vel[i, d] = (
                    w * vel[i, d]
                    + c1 * r1 * (p_best_pos[i, d] - pos[i, d])
                    + c2 * r2 * (g_best_pos[d] - pos[i, d])
                )
                # hız sınırını aşma durumunu kontrol et
                if vel[i, d] > v_max:
                    vel[i, d] = v_max
                elif vel[i, d] < -v_max:
                    vel[i, d] = -v_max

                # konum güncelle
                pos[i, d] += vel[i, d]

        best_value_history.append(g_best_score)

    best_solution = g_best_pos.tolist()
    best_value = g_best_score

    return best_solution, best_value, best_value_history
