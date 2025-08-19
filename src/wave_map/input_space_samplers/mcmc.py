import numpy as np
from typing import Optional, Callable, Dict, Any

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
        use_simulated_annealing: bool = False,
        initial_temp: float = 1.0,
        cooling_rate: float = 0.99,
        l2_threshold: Optional[float] = None
    ):
        """
        MCMC sampler for inverse inference with simulated annealing and per-dim adaptive proposals.

        Parameters
        ----------
        model : callable
            Function mapping input vector to output vector (1D array)
        proposal_scale : float or np.ndarray
            Initial proposal scale per dimension or scalar
        l2_threshold : float
            Optional L2 error threshold for early stopping
        """
        self.model = model
        self.burn_in = burn_in
        self.thin = thin
        self.likelihood_method = likelihood_method
        self.normalize_l2 = normalize_l2
        self.use_simulated_annealing = use_simulated_annealing
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.rng = np.random.default_rng(random_state)
        self.l2_threshold = l2_threshold

        if np.isscalar(proposal_scale):
            self.proposal_scale = float(proposal_scale)
        else:
            self.proposal_scale = np.array(proposal_scale, dtype=float)

    def _likelihood(self, proposed_output, target_output, sigma=0.05):
        """
        Compute log-likelihood of proposed_output matching target_output.
        
        Options:
          - "gaussian": standard squared-error likelihood
          - "l2": negative L2 norm (optionally normalized)
          - "cosine": directional similarity
          - "correlation": Pearson correlation
        """

        if self.likelihood_method == "gaussian":
            return -np.sum((proposed_output - target_output)**2) / (2 * sigma**2)
        
        elif self.likelihood_method == "l2":
            error = np.linalg.norm(proposed_output - target_output)
            if self.normalize_l2:
                error /= np.linalg.norm(target_output)
            return -error
    
        elif self.likelihood_method == "cosine":
            dot = np.dot(proposed_output, target_output)
            norm_prod = np.linalg.norm(proposed_output) * np.linalg.norm(target_output) + 1e-12
            similarity = dot / norm_prod
            return np.log((1 + similarity) / 2 + 1e-12)  # map to [0,1] then log
    
        elif self.likelihood_method == "correlation":
            p = proposed_output - np.mean(proposed_output)
            t = target_output - np.mean(target_output)
            corr = np.dot(p, t) / (np.linalg.norm(p) * np.linalg.norm(t) + 1e-12)
            return corr  # higher is better
    
        else:
            raise ValueError(f"Unknown likelihood method: {self.likelihood_method}")

    @staticmethod
    def _reflect_to_bounds(x: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        span = high - low
        span = np.where(span == 0, 1.0, span)
        y = (x - low) % (2.0 * span)
        y = np.where(y > span, 2.0 * span - y, y)
        return low + y

    def sample(
        self,
        target_output: np.ndarray,
        init_input: Optional[np.ndarray] = None,
        n_steps: int = 1000,
        sigma: float = 0.01,
        bounds: Optional[np.ndarray] = None,
        on_step: Optional[Callable[[int, np.ndarray, float], None]] = None
    ) -> Any:
        """
        Run MCMC to infer inputs given a target output.
        """
        dim = bounds.shape[0] if bounds is not None else len(init_input)
        
        # Optimized initialization
        if init_input is None:
            if bounds is not None:
                current_input = (bounds[:, 0] + bounds[:, 1]) / 2
            else:
                current_input = np.zeros(dim)
        else:
            current_input = np.array(init_input, dtype=float)

        # Proposal scales
        if np.isscalar(self.proposal_scale):
            proposal_scale = np.full(dim, self.proposal_scale)
        else:
            proposal_scale = self.proposal_scale.copy()
            if proposal_scale.shape[0] != dim:
                raise ValueError("proposal_scale must match input dimension")
        log_scale = np.log(proposal_scale)
        target_acc = 0.3
        adapt_start = max(50, dim*5)
        adapt_interval = 50
        adapt_until = n_steps // 2
        step_size = 0.1
        accepted_window = np.zeros(dim)
        proposed_window = np.zeros(dim)

        # Apply bounds
        if bounds is not None:
            low, high = bounds[:, 0], bounds[:, 1]
            current_input = self._reflect_to_bounds(current_input, low, high)

        current_output = self.model(current_input)
        current_like = self._likelihood(current_output, target_output, sigma)

        samples = np.zeros((n_steps, dim))
        likelihoods = np.zeros(n_steps)
        accept_count = 0
        temperature = self.initial_temp if self.use_simulated_annealing else 1.0

        for i in range(n_steps):
            current_scale = np.exp(log_scale)
            step = self.rng.normal(scale=current_scale, size=dim)
            proposal = current_input + step

            if bounds is not None:
                proposal = self._reflect_to_bounds(proposal, low, high)

            proposed_output = self.model(proposal)
            proposed_like = self._likelihood(proposed_output, target_output, sigma)

            log_alpha = (proposed_like - current_like) / temperature
            accepted = False
            if np.log(self.rng.random()) < log_alpha:
                current_input, current_output, current_like = proposal, proposed_output, proposed_like
                accept_count += 1
                accepted = True

            # Update adaptive per-dim proposal scales
            proposed_window += 1
            #accepted_window += accepted.astype(float)
            if accepted:
                accepted_window += 1.0  # add 1 to all dimensions

            if (i + 1) >= adapt_start and (i + 1) <= adapt_until and (i + 1) % adapt_interval == 0:
                acc_rate = accepted_window / np.maximum(1, proposed_window)
                log_scale += step_size * (acc_rate - target_acc)
                accepted_window[:] = 0
                proposed_window[:] = 0

            # Store
            samples[i] = current_input
            likelihoods[i] = current_like

            # Callback
            if on_step is not None:
                on_step(i, current_input, current_like)

            # Early stopping based on L2 error
            if self.l2_threshold is not None:
                l2_error = np.linalg.norm(current_output - target_output)
                if l2_error < self.l2_threshold:
                    break

            if self.use_simulated_annealing:
                temperature *= self.cooling_rate

        # Apply burn-in and thinning
        kept_samples = samples[self.burn_in::self.thin]
        kept_likes = likelihoods[self.burn_in::self.thin]
        best_idx = np.argmax(kept_likes)
        best_input = kept_samples[best_idx]
        best_like = kept_likes[best_idx]
        acceptance_rate = accept_count / n_steps

        return SimpleNamespace(
            samples=kept_samples,
            likelihoods=kept_likes,
            best_input=best_input,
            best_likelihood=best_like,
            acceptance_rate=acceptance_rate,
            final_temperature=temperature,
            final_proposal_scale=np.exp(log_scale)
        )


from types import SimpleNamespace
