import random
import numpy as np

def grey_wolf_optimizer(
    func,
    num_vars,
    bounds,
    pop_size=30,
    max_iter=50,
    a_start=2.0,
    a_end=0.0
):
    """
    Gelişmiş GWO (Grey Wolf Optimizer) Örneği:
      - Alpha, Beta, Delta konsepti
      - a parametresinin lineer azalması (a_start'tan a_end'e)
      - Sınır kontrolü (boyut boyut)
      - En iyi çözüm, en iyi değer ve iterasyon geçmişi döndürme
    """

    # bounds eğer tek sayı ise (ör. [-100, 100]),
    # her değişken için aynı alt/üst sınırı kullanalım;
    # yok eğer list of lists şeklinde geldiyse ( [ [lb1, ...], [ub1, ...] ] ),
    # o zaman doğrudan kullanalım.
    if not isinstance(bounds[0], list) and not isinstance(bounds[1], list):
        lb = [bounds[0]] * num_vars
        ub = [bounds[1]] * num_vars
    else:
        lb = bounds[0]
        ub = bounds[1]

    # Alpha, Beta, Delta kurt pozisyonları ve skorları
    Alpha_pos = [0.0] * num_vars
    Alpha_score = float("inf")

    Beta_pos = [0.0] * num_vars
    Beta_score = float("inf")

    Delta_pos = [0.0] * num_vars
    Delta_score = float("inf")

    # Popülasyon (kurtlar) başlangıcı
    wolves = []
    for _ in range(pop_size):
        wolf = [random.uniform(lb[d], ub[d]) for d in range(num_vars)]
        wolves.append(wolf)

    best_value_history = []  # Her iterasyonda Alpha skorunu saklayacağız

    for iteration in range(max_iter):
        # Tüm kurtların skorunu hesapla
        fitness = []
        for i in range(pop_size):
            # Sınır kontrolü
            for d in range(num_vars):
                wolves[i][d] = max(lb[d], min(ub[d], wolves[i][d]))

            fit_val = func(*wolves[i])
            fitness.append(fit_val)

            # Alpha, Beta, Delta güncelle
            if fit_val < Alpha_score:
                # En iyi, önceki Alpha'yı Beta'ya, önceki Beta'yı Delta'ya kaydır
                Delta_score, Delta_pos = Beta_score, Beta_pos[:]
                Beta_score, Beta_pos = Alpha_score, Alpha_pos[:]
                Alpha_score, Alpha_pos = fit_val, wolves[i][:]

            elif Alpha_score < fit_val < Beta_score:
                # İkinci en iyi -> Beta
                Delta_score, Delta_pos = Beta_score, Beta_pos[:]
                Beta_score, Beta_pos = fit_val, wolves[i][:]

            elif Beta_score < fit_val < Delta_score:
                # Üçüncü en iyi -> Delta
                Delta_score, Delta_pos = fit_val, wolves[i][:]

        # a parametresini lineer azaltma:
        # a, her iterasyonda a_start -> a_end (varsayılan 2.0 -> 0.0) aralığında azalacak
        if max_iter > 1:
            a = a_start - (a_start - a_end) * (iteration / (max_iter - 1))
        else:
            a = a_end

        # Her kurdu, Alpha-Beta-Delta'ya göre güncelle
        new_wolves = []
        for i in range(pop_size):
            current_wolf = wolves[i]
            new_wolf = []
            for d in range(num_vars):
                # Rastgele sayılar
                r1 = random.random()
                r2 = random.random()
                A1 = 2.0 * a * r1 - a
                C1 = 2.0 * r2

                # Alpha'ya göre güncelleme
                D_alpha = abs(C1 * Alpha_pos[d] - current_wolf[d])
                X1 = Alpha_pos[d] - A1 * D_alpha

                # Beta
                r1 = random.random()
                r2 = random.random()
                A2 = 2.0 * a * r1 - a
                C2 = 2.0 * r2
                D_beta = abs(C2 * Beta_pos[d] - current_wolf[d])
                X2 = Beta_pos[d] - A2 * D_beta

                # Delta
                r1 = random.random()
                r2 = random.random()
                A3 = 2.0 * a * r1 - a
                C3 = 2.0 * r2
                D_delta = abs(C3 * Delta_pos[d] - current_wolf[d])
                X3 = Delta_pos[d] - A3 * D_delta

                # Yeni konumun ortalaması
                new_dim = (X1 + X2 + X3) / 3.0

                # Sınırları koru
                new_dim = max(lb[d], min(ub[d], new_dim))
                new_wolf.append(new_dim)

            new_wolves.append(new_wolf)

        # Güncel popülasyon
        wolves = new_wolves

        # Bu iterasyonun en iyi değeri (Alpha_score)
        best_value_history.append(Alpha_score)

    # Döngü bittiğinde, Alpha_pos ve Alpha_score en iyi çözüm ve skor
    best_solution = Alpha_pos
    best_value = Alpha_score

    return best_solution, best_value, best_value_history
