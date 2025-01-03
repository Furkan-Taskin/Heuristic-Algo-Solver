import random
import numpy as np

def genetic_algorithm(
    func,
    num_vars,
    bounds,
    pop_size=30,
    max_iter=50,
    selection_type="tournament",
    crossover_type="1-point",
    crossover_rate=0.8,
    mutation_rate=0.1
):
    """
    Gelişmiş Genetik Algoritma Örneği:
      - Çoklu seçim metodu (tournament veya roulette)
      - Çoklu çaprazlama tipi (1-point, 2-point, uniform)
      - crossover_rate ve mutation_rate parametreleri
      - En iyi çözüm ve her iterasyondaki en iyi değerin geçmişi
    """

    # Sınırlar tek sayı şeklindeyse, her değişken için aynı değeri kullan
    if not isinstance(bounds[0], list) and not isinstance(bounds[1], list):
        lb = [bounds[0]] * num_vars
        ub = [bounds[1]] * num_vars
    else:
        # bounds, [ [lb1, lb2, ...], [ub1, ub2, ...] ] gibi verilmişse
        lb = bounds[0]
        ub = bounds[1]

    # Rastgele popülasyon oluştur
    population = []
    for _ in range(pop_size):
        indiv = [random.uniform(lb[i], ub[i]) for i in range(num_vars)]
        population.append(indiv)

    best_value_history = []

    # Seçim fonksiyonları
    def roulette_wheel_select(pop, fitness_values):
        """Rulet tekerleği seçimi."""
        total_fitness = sum(fitness_values)
        # Maksimize değil minimize ediyorsak fitness değerlerini tersine çevir
        # istersen "1 / (1 + f)" gibi dönüştürme yapabilirsin; 
        # örnekte basitlik için doğrudan kullanıyoruz (yaklaşım farkı olabilir).
        pick = random.uniform(0, total_fitness)
        current = 0
        for i, f in enumerate(fitness_values):
            current += f
            if current > pick:
                return pop[i]
        return pop[-1]  # Teorik olarak buraya nadiren düşülür

    def tournament_select(pop, fitness_values, t_size=3, win_prob=0.75):
        """Turnuva seçimi."""
        selected_indices = random.sample(range(len(pop)), t_size)
        # fitness değerleri küçük olan daha iyi ise sıralamayı ters çevirme ya da 
        # minimizasyon için adaptasyon yap
        # Burada fonksiyon değeri küçük olan en iyi, o yüzden sort(key=...).
        selected = sorted(selected_indices, key=lambda idx: fitness_values[idx])
        # Win prob ile en iyi seç, aksi halde diğerlerinden biri
        if random.random() < win_prob:
            return pop[selected[0]]
        else:
            return pop[random.choice(selected[1:])]

    def select_parent(pop, fitness_values):
        """Seçim tipine göre ebeveyn seç."""
        if selection_type == "roulette_wheel":
            return roulette_wheel_select(pop, fitness_values)
        elif selection_type == "tournament":
            return tournament_select(pop, fitness_values)
        else:
            # Default turnuva seçimi
            return tournament_select(pop, fitness_values)

    # Çaprazlama fonksiyonu
    def crossover(p1, p2):
        """crossover_type parametresine göre çocuk üret."""
        if crossover_type == "1-point":
            if num_vars > 1:
                point = random.randint(1, num_vars - 1)
                c1 = p1[:point] + p2[point:]
                c2 = p2[:point] + p1[point:]
            else:
                # Tek değişken varsa çaprazlamanın anlamı yok
                c1, c2 = p1[:], p2[:]
        elif crossover_type == "2-point":
            if num_vars > 2:
                points = sorted(random.sample(range(1, num_vars), 2))
                c1 = (p1[:points[0]] + p2[points[0]:points[1]] + p1[points[1]:])
                c2 = (p2[:points[0]] + p1[points[0]:points[1]] + p2[points[1]:])
            else:
                c1, c2 = p1[:], p2[:]
        elif crossover_type == "uniform":
            c1, c2 = [], []
            for i in range(num_vars):
                if random.random() < 0.5:
                    c1.append(p1[i])
                    c2.append(p2[i])
                else:
                    c1.append(p2[i])
                    c2.append(p1[i])
        else:
            # Varsayılan 1-point
            if num_vars > 1:
                point = random.randint(1, num_vars - 1)
                c1 = p1[:point] + p2[point:]
                c2 = p2[:point] + p1[point:]
            else:
                c1, c2 = p1[:], p2[:]
        return c1, c2

    # Mutasyon fonksiyonu
    def mutate(child):
        """Belirlenen mutation_rate'e göre çocuğu mutasyona uğrat."""
        for i in range(num_vars):
            if random.random() < mutation_rate:
                # Küçük bir rastgele ekleme/çıkarma yapıyoruz
                # Dilersen Gaussian gibi farklı mutasyon yaklaşımı da deneyebilirsin
                mutation_step = random.uniform(-1, 1)
                child[i] += mutation_step
                # Sınırları kontrol et
                child[i] = max(lb[i], min(child[i], ub[i]))
        return child

    for it in range(max_iter):
        # Fitnes değerlerini hesapla (minimizasyon problemine göre)
        fitness = [func(*indiv) for indiv in population]

        # En iyi bireyi bul
        best_idx = np.argmin(fitness)
        best_val = fitness[best_idx]
        best_value_history.append(best_val)

        new_population = []

        # En iyi bireyi doğrudan sonraki nesle aktarma (elitizm)
        best_individual = population[best_idx][:]
        new_population.append(best_individual)

        # Popülasyonu eşit sayıda çift olarak (crossover için) doldur
        while len(new_population) < pop_size:
            # Ebeveyn seçimi
            parent1 = select_parent(population, fitness)
            parent2 = select_parent(population, fitness)

            # Çaprazlama olasılığı
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]

            # Mutasyon
            child1 = mutate(child1)
            child2 = mutate(child2)

            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        # Yeni nesil popülasyona geç
        population = new_population

    # Son iterasyonun fitnes değerlerini hesapla
    final_fitness = [func(*indiv) for indiv in population]
    best_idx = np.argmin(final_fitness)
    best_solution = population[best_idx]
    best_value = final_fitness[best_idx]

    return best_solution, best_value, best_value_history
