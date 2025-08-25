import numpy as np
from typing import Optional, Callable, Any
from types import SimpleNamespace

class McmcSampler:
    def __init__(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        proposal_scale: float = 0.1,
        burn_in: int = 0,
        thin: int = 1,
        likelihood_method: str = "gaussian",
        normalize_l2: bool = True,
        random_state: Optional[int] = None,
        verbose: bool = True,
        # --- Simulated annealing ---
        use_simulated_annealing: bool = False,
        initial_temp: float = 1.0,
        cooling_rate: float = 0.99,
        # --- Mixture proposals ---
        global_step_prob: float = 0.1,         # prob. of using a large "global" Gaussian step
        independent_jump_prob: float = 0.0,    # prob. of uniform draw in bounds (requires bounds)
        local_scale: Optional[float | np.ndarray] = None,
        global_scale: Optional[float | np.ndarray] = None,
    ):
        """
        Metropolis MCMC with simulated annealing and mixture proposals.

        Mixture proposals:
          - With prob independent_jump_prob (if bounds provided): propose x ~ Uniform(bounds)
          - Else with prob global_step_prob: large Gaussian step
          - Else: local Gaussian step
        """
        self.model = model
        self.burn_in = burn_in
        self.thin = thin
        self.likelihood_method = likelihood_method
        self.normalize_l2 = normalize_l2
        self.rng = np.random.default_rng(random_state)
        self.verbose = verbose

        # Annealing
        self.use_simulated_annealing = use_simulated_annealing
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate

        # Base proposal scale
        if np.isscalar(proposal_scale):
            self.proposal_scale = float(proposal_scale)
        else:
            self.proposal_scale = np.array(proposal_scale, dtype=float)

        # Mixture proposal settings
        self.global_step_prob = float(global_step_prob)
        self.independent_jump_prob = float(independent_jump_prob)

        self.local_scale = local_scale  # if None, set in sample() based on proposal_scale
        self.global_scale = global_scale  # if None, set in sample() as 5x local_scale by default

    def _likelihood(self, proposed_output, target_output, sigma=0.05):
        """Compute log-likelihood of proposed_output matching target_output."""
        if self.likelihood_method == "gaussian":
            return -np.sum((proposed_output - target_output)**2) / (2 * sigma**2)
        
        elif self.likelihood_method == "l2":
            error = np.linalg.norm(proposed_output - target_output)
            if self.normalize_l2:
                denom = np.linalg.norm(target_output)
                if denom > 0:
                    error /= denom
            return -error / sigma
    
        elif self.likelihood_method == "cosine":
            dot = np.dot(proposed_output, target_output)
            norm_prod = np.linalg.norm(proposed_output) * np.linalg.norm(target_output) + 1e-12
            similarity = dot / norm_prod
            return np.log((1 + similarity) / 2 + 1e-12)
    
        elif self.likelihood_method == "correlation":
            p = proposed_output - np.mean(proposed_output)
            t = target_output - np.mean(target_output)
            corr = np.dot(p, t) / (np.linalg.norm(p) * np.linalg.norm(t) + 1e-12)
            return corr
    
        else:
            raise ValueError(f"Unknown likelihood method: {self.likelihood_method}")

    @staticmethod
    def _reflect_to_bounds(x: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        span = high - low
        span = np.where(span == 0, 1.0, span)
        y = (x - low) % (2.0 * span)
        y = np.where(y > span, 2.0 * span - y, y)
        return low + y

    def _draw_proposal(
        self,
        current_input: np.ndarray,
        low: Optional[np.ndarray],
        high: Optional[np.ndarray],
        local_scale_vec: np.ndarray,
        global_scale_vec: np.ndarray,
    ) -> np.ndarray:
        """Mixture proposal: independent jump (if bounds), global step, or local step."""
        u = self.rng.random()
        if (low is not None) and (high is not None) and (u < self.independent_jump_prob):
            # Independent uniform draw within bounds
            proposal = self.rng.uniform(low=low, high=high)
            return proposal

        # Choose global vs local Gaussian step
        if u < self.independent_jump_prob + self.global_step_prob:
            step = self.rng.normal(scale=global_scale_vec, size=current_input.size)
        else:
            step = self.rng.normal(scale=local_scale_vec, size=current_input.size)

        return current_input + step

    def sample(
        self,
        target_output: np.ndarray,
        init_input: Optional[np.ndarray] = None,
        n_steps: int = 1000,
        sigma: float = 0.01,
        bounds: Optional[np.ndarray] = None
    ) -> Any:
        """Run Metropolis MCMC with simulated annealing + mixture proposals."""
        dim = bounds.shape[0] if bounds is not None else len(init_input)
        
        # Initialization
        if init_input is None:
            if bounds is not None:
                current_input = (bounds[:, 0] + bounds[:, 1]) / 2
            else:
                current_input = np.zeros(dim)
        else:
            current_input = np.array(init_input, dtype=float)

        # Proposal scales (vectorized)
        base = self.proposal_scale
        if np.isscalar(base):
            base_vec = np.full(dim, base, dtype=float)
        else:
            if len(base) != dim:
                raise ValueError("proposal_scale must match input dimension")
            base_vec = np.array(base, dtype=float)

        if self.local_scale is None:
            local_scale_vec = base_vec
        else:
            ls = self.local_scale
            local_scale_vec = np.full(dim, ls, dtype=float) if np.isscalar(ls) else np.array(ls, dtype=float)

        if self.global_scale is None:
            global_scale_vec = 5.0 * local_scale_vec  # default: 5x larger jumps
        else:
            gs = self.global_scale
            global_scale_vec = np.full(dim, gs, dtype=float) if np.isscalar(gs) else np.array(gs, dtype=float)

        # Bounds
        low = high = None
        if bounds is not None:
            low, high = bounds[:, 0].astype(float), bounds[:, 1].astype(float)
            current_input = self._reflect_to_bounds(current_input, low, high)

        # Start chain
        current_output = self.model(current_input)
        current_like = self._likelihood(current_output, target_output, sigma)

        samples = np.zeros((n_steps, dim))
        likelihoods = np.zeros(n_steps)
        accept_count = 0

        best_input = current_input.copy()
        best_l2 = np.linalg.norm(current_output - target_output)

        # Temperature
        temperature = self.initial_temp if self.use_simulated_annealing else 1.0

        for i in range(n_steps):
            # Mixture proposal
            proposal = self._draw_proposal(
                current_input=current_input,
                low=low, high=high,
                local_scale_vec=local_scale_vec,
                global_scale_vec=global_scale_vec
            )

            # Apply bounds (reflection)
            if low is not None:
                proposal = self._reflect_to_bounds(proposal, low, high)

            # Optional ordering constraint (keep if you need it)
            proposal[2:5] = np.sort(proposal[2:5])

            # Evaluate
            proposed_output = self.model(proposal)
            proposed_like = self._likelihood(proposed_output, target_output, sigma)

            # Annealed Metropolis criterion
            log_alpha = (proposed_like - current_like) / temperature
            if np.log(self.rng.random()) < log_alpha:
                current_input, current_output, current_like = proposal, proposed_output, proposed_like
                accept_count += 1

            samples[i] = current_input
            likelihoods[i] = current_like

            # Track best
            l2_error = np.linalg.norm(current_output - target_output)
            if l2_error < best_l2:
                best_l2 = l2_error
                best_input = current_input.copy()

            # Cooling
            if self.use_simulated_annealing:
                temperature *= self.cooling_rate

            if self.verbose and (i + 1) % 10 == 0:
                acc_rate = accept_count / (i + 1)
                best_str = np.array2string(best_input, precision=4, separator=',', suppress_small=True)
                #true_str = np.array2string(true_input, precision=4, separator=',', suppress_small=True)

            
                # Define the block of text we want to print
                block = (
                    f"Best input: {best_str}\n"
                    f"Step {i+1}/{n_steps} | Best L2: {best_l2:.6f}\n"
                    f"Acc: {acc_rate:.3f} | Temp: {temperature:.4f}\n"
                    f"p_global: {self.global_step_prob:.2f} | "
                    f"p_indep: {self.independent_jump_prob:.2f}\n"
                )
            
                # Count how many lines the block has
                n_lines = block.count("\n")
            
                # Move cursor up and clear old block
                print(f"\033[{n_lines}A\033[J", end="")
            
                # Print new block
                print(block, end="", flush=True)



        kept_samples = samples[self.burn_in::self.thin]
        kept_likes = likelihoods[self.burn_in::self.thin]

        acceptance_rate = accept_count / n_steps

        if self.verbose:
            print("\nSampling complete.")

        return SimpleNamespace(
            samples=kept_samples,
            likelihoods=kept_likes,
            best_input=best_input,
            best_l2=best_l2,
            acceptance_rate=acceptance_rate,
            final_temperature=temperature
        )
