# wave_map/input_space_samplers/mcmc.py
import numpy as np

class McmcSampler:
    def __init__(self, model, proposal_scale=0.1, burn_in=0, thin=1,
                 likelihood_method="l2", normalize_l2=True, random_state=None):
        """
        MCMC sampler for inverse inference.

        Parameters
        ----------
        model : callable
            Function mapping input parameters -> predicted output.
        proposal_scale : float or np.ndarray
            Step size(s) for proposal distribution. If array, must match input dimension.
        burn_in : int
            Number of initial samples to discard.
        thin : int
            Keep only every `thin`-th sample after burn-in.
        likelihood_method : str
            "gaussian" or "l2" likelihood.
        normalize_l2 : bool
            If True, normalize L2 norm by ||target_output||.
        random_state : int or None
            Seed for RNG.
        """
        self.model = model
        self.proposal_scale = proposal_scale
        self.burn_in = burn_in
        self.thin = thin
        self.likelihood_method = likelihood_method
        self.normalize_l2 = normalize_l2
        self.rng = np.random.default_rng(random_state)

    def _likelihood(self, proposed_output, target_output, sigma=0.05):
        if self.likelihood_method == "gaussian":
            return -np.sum((proposed_output - target_output)**2) / (2 * sigma**2)
        elif self.likelihood_method == "l2":
            error = np.linalg.norm(proposed_output - target_output)
            if self.normalize_l2:
                error /= np.linalg.norm(target_output)
            return -error
        else:
            raise ValueError(f"Unknown likelihood method: {self.likelihood_method}")

    def sample(self, target_output, init_input, n_steps=1000, sigma=0.01, bounds=None, on_step=None):
        """
        Run MCMC to infer inputs given a target output.

        Parameters
        ----------
        target_output : np.ndarray
            The output vector we want to match.
        init_input : np.ndarray
            Initial guess for input vector.
        n_steps : int
            Total number of MCMC steps.
        sigma : float
            Used only for Gaussian likelihood.
        bounds : np.ndarray of shape (dim, 2)
            Lower and upper bounds for each input dimension.
        on_step : callable
            Optional callback: on_step(step_idx, current_input, current_likelihood)
        """
        init_input = np.array(init_input, dtype=float)
        dim = len(init_input)

        # Allow per-dimension proposal scales
        if np.isscalar(self.proposal_scale):
            proposal_scale = np.full(dim, self.proposal_scale)
        else:
            proposal_scale = np.array(self.proposal_scale, dtype=float)
            if proposal_scale.shape[0] != dim:
                raise ValueError("proposal_scale must match input dimension")

        current_input = init_input.copy()
        current_output = self.model(current_input)
        current_like = self._likelihood(current_output, target_output, sigma)

        samples = np.zeros((n_steps, dim))
        likelihoods = np.zeros(n_steps)
        accept_count = 0

        for i in range(n_steps):
            step = self.rng.normal(scale=proposal_scale, size=dim)
            proposal = current_input + step

            if bounds is not None:
                proposal = np.clip(proposal, bounds[:, 0], bounds[:, 1])

            proposed_output = self.model(proposal)
            proposed_like = self._likelihood(proposed_output, target_output, sigma)

            # Metropolis-Hastings acceptance
            log_alpha = proposed_like - current_like
            if np.log(self.rng.random()) < log_alpha:
                current_input, current_output, current_like = proposal, proposed_output, proposed_like
                accept_count += 1

            samples[i] = current_input
            likelihoods[i] = current_like

            if on_step is not None:
                on_step(i, current_input, current_like)

        # Apply burn-in and thinning
        kept_samples = samples[self.burn_in::self.thin]
        kept_likes = likelihoods[self.burn_in::self.thin]

        # Best sample
        best_idx = np.argmax(kept_likes)
        best_input = kept_samples[best_idx]
        best_like = kept_likes[best_idx]

        acceptance_rate = accept_count / n_steps

        return {
            "samples": kept_samples,
            "likelihoods": kept_likes,
            "best_input": best_input,
            "best_likelihood": best_like,
            "acceptance_rate": acceptance_rate,
        }
