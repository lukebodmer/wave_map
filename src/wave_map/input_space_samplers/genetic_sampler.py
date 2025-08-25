import numpy as np
from dataclasses import dataclass

@dataclass
class GAResult:
    best_input: np.ndarray
    best_fitness: float
    history: list  # best fitness each generation


class GeneticSampler:
    def __init__(self, model, bounds, population_size=100, n_generations=200,
                 mutation_rate=0.1, crossover_rate=0.8, random_state=None,
                 mutation_scales=None, fitness_type="l2"):
        """
        Parameters
        ----------
        model : callable
            Predictive model f(x) returning output vector.
        bounds : np.ndarray
            Array of shape (n_params, 2), parameter lower/upper bounds.
        population_size : int
            Number of individuals per generation.
        n_generations : int
            Number of generations to evolve.
        mutation_rate : float
            Probability of mutating a gene.
        crossover_rate : float
            Probability of crossover between two parents.
        random_state : int or None
            RNG seed.
        mutation_scales : array-like or None
            Per-gene mutation scaling factors.
        fitness_type : str
            "l2" = plain L2 norm
            "nl2" = normalized L2 norm
            "corr" = 1 - correlation coefficient
            "hybrid" = 0.5*NL2 + 0.5*(1 - corr)
        """
        self.model = model
        self.bounds = np.asarray(bounds, dtype=float)
        self.pop_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.rng = np.random.default_rng(random_state)
        self.n_genes = self.bounds.shape[0]

        # per-gene mutation scales
        if mutation_scales is None:
            self.mutation_scales = np.ones(self.n_genes)
        else:
            self.mutation_scales = np.asarray(mutation_scales, dtype=float)

        self.fitness_type = fitness_type.lower()

    def _fitness(self, individual, target_output):
        pred = self.model(individual)
        if self.fitness_type == "l2":
            return np.linalg.norm(pred - target_output)
        elif self.fitness_type == "nl2":
            return np.linalg.norm(pred - target_output) / (np.linalg.norm(target_output) + 1e-12)
        elif self.fitness_type == "corr":
            if np.std(pred) < 1e-12 or np.std(target_output) < 1e-12:
                return 1.0  # avoid divide-by-zero
            return 1.0 - np.corrcoef(pred, target_output)[0, 1]
        elif self.fitness_type == "hybrid":
            nl2 = np.linalg.norm(pred - target_output) / (np.linalg.norm(target_output) + 1e-12)
            corr = 1.0 - np.corrcoef(pred, target_output)[0, 1] if np.std(pred) > 1e-12 else 1.0
            return 0.5 * nl2 + 0.5 * corr
        else:
            raise ValueError(f"Unknown fitness_type: {self.fitness_type}")

    def _init_population(self):
        """Initialize population uniformly within bounds."""
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        return self.rng.uniform(low, high, size=(self.pop_size, self.n_genes))

    def _tournament_selection(self, population, fitness, k=3):
        """Select one parent using tournament selection."""
        idxs = self.rng.choice(len(population), size=k, replace=False)
        best_idx = idxs[np.argmin(fitness[idxs])]
        return population[best_idx]

    def _crossover(self, parent1, parent2):
        """Arithmetic crossover between two parents."""
        if self.rng.random() < self.crossover_rate:
            alpha = self.rng.random()
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = alpha * parent2 + (1 - alpha) * parent1
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()

    def _mutate(self, individual):
        """Mutate genes with Gaussian noise using per-gene mutation scales."""
        for i in range(self.n_genes):
            if self.rng.random() < self.mutation_rate:
                span = self.bounds[i, 1] - self.bounds[i, 0]
                noise = self.rng.normal(0, self.mutation_scales[i] * span)
                individual[i] += noise
                # Clamp to bounds
                individual[i] = np.clip(individual[i], self.bounds[i, 0], self.bounds[i, 1])
        return individual


    def evolve(self, target_output, n_elite=5):
        """Run the genetic algorithm with elitism to approximate inverse input for target_output."""
        population = self._init_population()
        fitness = np.array([self._fitness(ind, target_output) for ind in population])

        best_idx = np.argmin(fitness)
        best_input = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        history = [best_fitness]

        for gen in range(self.n_generations):
            # --- sort population by fitness (ascending) ---
            sorted_idx = np.argsort(fitness)
            elites = population[sorted_idx[:n_elite]].copy()
            elite_fitness = fitness[sorted_idx[:n_elite]].copy()

            # --- create new population ---
            new_population = elites.tolist()  # start with elites

            while len(new_population) < self.pop_size:
                # Parent selection
                p1 = self._tournament_selection(population, fitness)
                p2 = self._tournament_selection(population, fitness)
                # Crossover
                c1, c2 = self._crossover(p1, p2)
                # Mutation
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                new_population.extend([c1, c2])

            population = np.array(new_population[:self.pop_size])
            fitness = np.array([self._fitness(ind, target_output) for ind in population])

            # Update best solution
            gen_best_idx = np.argmin(fitness)
            gen_best_fitness = fitness[gen_best_idx]
            if gen_best_fitness < best_fitness:
                best_fitness = gen_best_fitness
                best_input = population[gen_best_idx].copy()

            history.append(best_fitness)

            # --- Real-time printing ---
            best_str = np.array2string(best_input, precision=4, separator=',', suppress_small=True)
            print(f"Gen {gen+1:4d}/{self.n_generations} | Best L2: {best_fitness:.6f} | Best guess: {best_str}", 
                  end='\r', flush=True)

        print()  # final newline
        return GAResult(best_input=best_input, best_fitness=best_fitness, history=history)
