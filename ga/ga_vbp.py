import os
import random
import time
import numpy as np
from collections import defaultdict

# inst loading

def load_instance(path):
    with open(path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    first = lines[0].split()
    if len(first) == 1:
        d = int(first[0])
        capacities = list(map(float, lines[1].split()))
        n = int(lines[2])
        raw = lines[3:3+n]
    else:
        n, d = map(int, first)
        capacities = list(map(float, lines[1].split()))
        raw = lines[2:2+n]
    items = []
    for line in raw:
        parts = list(map(float, line.split()))
        vec = parts[:-1] if len(parts) == d + 1 and parts[-1] == 1 else parts
        items.append(vec)
    return n, d, np.array(capacities), np.array(items)

# bf/ff heuristike

def first_fit(items, capacities):
    n, d = items.shape
    solution = [-1]*n
    loads = []
    for i, vec in enumerate(items):
        placed = False
        for b, load in enumerate(loads):
            if np.all(load + vec <= capacities):
                solution[i] = b
                loads[b] += vec
                placed = True
                break
        if not placed:
            solution[i] = len(loads)
            loads.append(vec.copy())
    return solution

def best_fit(items, capacities):
    n, d = items.shape
    solution = [-1] * n
    bin_loads = []
    for i, item in enumerate(items):
        best_bin = -1
        best_residual = float('inf')
        for b, load in enumerate(bin_loads):
            new_load = load + item
            if np.all(new_load <= capacities):
                residual = np.sum(capacities - new_load)
                if residual < best_residual:
                    best_residual = residual
                    best_bin = b
        if best_bin == -1:
            solution[i] = len(bin_loads)
            bin_loads.append(item.copy())
        else:
            solution[i] = best_bin
            bin_loads[best_bin] += item
    return solution

# (de)kodiranje

def decode(solution):
    bins = defaultdict(list)
    for idx, b in enumerate(solution):
        bins[b].append(idx)
    return bins

# kesirani fitnes za brze izracunavanjae

def create_cached_fitness(items, capacities):
    cache = {}
    def cached_fitness(solution):
        key = tuple(solution)
        if key in cache:
            return cache[key]
        bins = decode(solution)
        penalty = 0
        for idxs in bins.values():
            load = items[idxs].sum(axis=0)
            excess = np.maximum(0, load - capacities)
            penalty += np.sum((excess / capacities) ** 2)
        score = len(bins) + penalty  # treba pogledati penalizaciju, mislim da bude stuck u locmin
        cache[key] = score
        return score
    return cached_fitness

# reparacija -- todo repairaj sa bf/ff solo

def repair(solution, items, capacities):
    bins = decode(solution)
    new_solution = [-1] * len(solution)
    new_bins = []
    for idxs in bins.values():
        load = np.zeros_like(capacities)
        current_bin = []
        for i in idxs:
            if np.all(load + items[i] <= capacities):
                current_bin.append(i)
                load += items[i]
            else:
                new_bins.append([i])
        if current_bin:
            new_bins.append(current_bin)
    for b_id, idxs in enumerate(new_bins):
        for i in idxs:
            new_solution[i] = b_id
    return new_solution

# selekcija

def tournament_selection(pop, scores, k=3):
    candidates = random.sample(list(zip(pop, scores)), k)
    return min(candidates, key=lambda x: x[1])[0]

# ukrstanje

def one_point_crossover(p1, p2):
    pt = random.randint(1, len(p1)-1)
    return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]

#mutop

def mutate(solution):
    mutation_rate = 0.2
    max_bin_id = max(solution) + 1
    for i in range(len(solution)):
        if random.random() < mutation_rate:
            solution[i] = random.randint(0, max_bin_id)

# ga

def genetic_algorithm(items, capacities, time_limit, pop_size, pc, pm):
    start = time.time()
    max_generations = 1000
    fitness_fn = create_cached_fitness(items, capacities)
    n = len(items)

    # init sa heuristikama
    pop = []
    for _ in range(int(pop_size * 0.4)):
        pop.append(first_fit(items, capacities))
    for _ in range(int(pop_size * 0.4)):
        pop.append(best_fit(items, capacities))
    for _ in range(pop_size - len(pop)):
        sol = [random.randint(0, n // 2) for _ in range(n)]
        pop.append(sol)

    best_sol = None
    best_score = float('inf')
    generation = 0
    stagnation_counter = 0

    while time.time() - start < time_limit and generation < max_generations:
        scores = [fitness_fn(sol) for sol in pop]
        best_idx = np.argmin(scores)
        if scores[best_idx] < best_score:
            best_score = scores[best_idx]
            best_sol = pop[best_idx].copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        # restart ako stagnira
        if stagnation_counter >= 50:
            elites = sorted(zip(pop, scores), key=lambda x: x[1])[:10]
            pop = []
            for _ in range(pop_size - len(elites)):
                if random.random() < 0.5:
                    pop.append(first_fit(items, capacities))
                else:
                    pop.append(best_fit(items, capacities))
            pop.extend([e[0].copy() for e in elites])
            stagnation_counter = 0

        # nova generacija
        new_pop = [best_sol.copy()]
        while len(new_pop) < pop_size:
            p1 = tournament_selection(pop, scores)
            p2 = tournament_selection(pop, scores)
            if random.random() < pc:
                c1, c2 = one_point_crossover(p1.copy(), p2.copy())
            else:
                c1, c2 = p1.copy(), p2.copy()
            if random.random() < pm:
                mutate(c1)
            if random.random() < pm:
                mutate(c2)
            c1 = repair(c1, items, capacities)
            c2 = repair(c2, items, capacities)
            new_pop.extend([c1, c2])
        pop = new_pop[:pop_size]
        generation += 1

    return best_sol, best_score, time.time() - start


# main dio

if __name__ == '__main__':
    in_folder = 'normirane_inst'
    out_folder = 'rezultati_ga'
    os.makedirs(out_folder, exist_ok=True)
    runs_per_instance = 5
    time_limit = 600  

    for fname in os.listdir(in_folder):
        if not fname.endswith('.vbp'):
            continue
        path = os.path.join(in_folder, fname)
        n, d, capacities, items = load_instance(path)
        results = []
        times = []
        solutions = []
        print(f"Instanca {fname}, pokrećem {runs_per_instance} puta...")

        for run in range(runs_per_instance):
            sol, sc, elapsed = genetic_algorithm(
                items, capacities, time_limit, pop_size=100, pc=0.9, pm=0.3
            )
            bins_used = len(decode(sol))
            results.append(bins_used)
            times.append(elapsed)
            solutions.append(sol)
            print(f" Run {run+1}: bins={bins_used}, time={elapsed:.2f}s")

        avg_bins = sum(results) / runs_per_instance
        avg_time = sum(times) / runs_per_instance
        with open(os.path.join(out_folder, fname.replace('.vbp', '_summary.txt')), 'w') as f:
            f.write(f"Instance: {fname}\nRuns: {runs_per_instance}\n")
            f.write(f"Average bins: {avg_bins}\n")
            f.write(f"Average time: {avg_time:.2f}s\n\n")
            for i, (b, sol, t) in enumerate(zip(results, solutions, times), 1):
                f.write(f"Run {i}: bins={b}, time={t:.2f}s\nSolution: {' '.join(map(str, sol))}\n\n")
        print(f"Done {fname}: avg bins={avg_bins}, avg time={avg_time:.2f}s\n")
