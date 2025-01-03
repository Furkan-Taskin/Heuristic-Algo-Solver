import random
import numpy as np

def big_bang_big_crunch(
    func,
    num_vars,
    bounds,
    pop_size=30,
    max_iter=50,
    shrink_factor=0.9,
    elite_count=1
):
    """
    Gelişmiş BB-BC (Big Bang Big Crunch) algoritması:
      - Kütle merkezi (1/fitness) tabanlı veya basit ortalama
      - Her iterasyonda yarıçapın shrink_factor ile azalması (monotonik azalma)
      - Elitizm: En iyi birey(ler) yeni nesilde korunur
      - En iyi çözüm, en iyi değer ve iterasyon geçmişi döndürme
    """

    # bounds [lb, ub] ya da [[lb1, lb2,..],[ub1, ub2,..]]
    if not isinstance(bounds[0], list) and not isinstance(bounds[1], list):
        lb = np.array([bounds[0]] * num_vars, dtype=float)
        ub = np.array([bounds[1]] * num_vars, dtype=float)
    else:
        lb = np.array(bounds[0], dtype=float)
        ub = np.array(bounds[1], dtype=float)

    # Başlangıç popülasyonu
    population = np.random.uniform(low=lb, high=ub, size=(pop_size, num_vars))

    # Yarıçap = (ub - lb) * 0.5 (her boyutta), ilk patlama aralığı
    base_radius = (ub - lb) * 0.5

    # Iterasyonlar boyunca en iyi fitnesları tutalım
    best_value_history = []

    # Elitizm için en iyi bireylerin saklanacağı yer
    elite_count = min(elite_count, pop_size)  # Güvenlik amacıyla

    for iteration in range(max_iter):
        # 1) Fitness hesapla
        fitness = np.array([func(*indiv) for indiv in population])

        # 2) En iyi bireyleri sırala
        sort_indices = np.argsort(fitness)
        best_indices = sort_indices[:elite_count]  # en iyi elite_count sayısı
        worst_indices = sort_indices[elite_count:] # geri kalanlar

        # 3) En iyi değer ve bireyi kaydet
        best_val = fitness[sort_indices[0]]
        best_sol = population[sort_indices[0]].copy()
        best_value_history.append(best_val)

        # 4) Kütle merkezi hesabı
        # Burada 1/fitness tabanlı ağırlık kullanıyoruz
        # Not: fitness çok küçük veya negatif olabilir, eps vb. ekleyerek düzeltme sağlanabilir
        eps = 1e-12
        inv_fit = 1.0 / (fitness + eps)
        total_inv = np.sum(inv_fit)
        weights = inv_fit / (total_inv + eps)
        mass_center = np.sum(population * weights.reshape(-1, 1), axis=0)

        # Alternatif: Basit ortalama
        # mass_center = np.mean(population, axis=0)

        # 5) Yarıçapın azaltılması (monotonik)
        # Örnekte üssel bir azalma kullanılıyor.
        # iteration=0 -> radius=base_radius*(shrink_factor^1)
        radius = base_radius * (shrink_factor ** (iteration + 1))

        # 6) Yeni popülasyon oluştur
        new_population = []
        # Elit bireyleri doğrudan ekle (elitizm)
        for idx in best_indices:
            new_population.append(population[idx].copy())

        # Kalan pop_size - elite_count bireyleri Big Bang etrafında oluştur
        needed = pop_size - elite_count
        offsets = np.random.uniform(low=-radius, high=radius, size=(needed, num_vars))
        for i in range(needed):
            candidate = mass_center + offsets[i]
            # Sınırları koru
            candidate = np.clip(candidate, lb, ub)
            new_population.append(candidate)

        # Array'e dönüştür
        population = np.array(new_population)

    # Döngü bitince son popülasyonda en iyi birey
    fitness = np.array([func(*indiv) for indiv in population])
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_value = fitness[best_idx]

    return best_solution.tolist(), best_value, best_value_history
