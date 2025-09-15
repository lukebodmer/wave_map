import torch
from torch.distributions.normal import Normal
from torch.distributions.studentT import StudentT
from logging import getLogger


class ParallelPartialEmulator:
    def __init__(self, design, response, nugget=1e-6, a=0.2,
                 device=None, method="post_mode", prior_choice="ref_approx"):
        """
        design  : np.ndarray (n, p)
        response: np.ndarray (n, k)
        """
        self.logger = getLogger("emulatorlog")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.X = torch.as_tensor(design, dtype=torch.float64, device=self.device)
        self.Y = torch.as_tensor(response, dtype=torch.float64, device=self.device)

        self.n, self.p = self.X.shape
        self.k = self.Y.shape[1]

        self.nugget = float(nugget)
        self.a = a
        self.b = 1 / (self.n ** (1 / self.p)) * (self.a + self.p)

        self.method = method
        self.prior_choice = prior_choice

        # kernel fixed to Matern 5/2 for now
        self.alpha = torch.full((self.p,), 2.5, dtype=torch.float64, device=self.device)
        self.trend = torch.ones((self.n, 1), dtype=torch.float64, device=self.device)

        # fitted quantities
        self.beta_hat = None
        self.theta_hat = None
        self.sigma2_hat = None
        self.L = None

    # --------------------
    # Kernel utils
    # --------------------
    def _pairwise_distances(self):
        R0 = []
        for j in range(self.p):
            diff = torch.abs(self.X[:, j][:, None] - self.X[:, j])
            R0.append(diff)
        return R0

    def _kernel_matrix(self, beta, R0, nugget=None):
        """Matern 5/2 correlation matrix with optional nugget override."""
        R = torch.ones((self.n, self.n), dtype=torch.float64, device=self.device)
        for j in range(self.p):
            r = R0[j] * beta[j]
            sqrt5_r = torch.sqrt(torch.tensor(5.0, dtype=torch.float64)) * r
            R *= (1 + sqrt5_r + 5 * r**2 / 3.0) * torch.exp(-sqrt5_r)
        nugget_val = self.nugget if nugget is None else nugget
        R += nugget_val * torch.eye(self.n, dtype=torch.float64, device=self.device)
        return R

    # --------------------
    # Objectives
    # --------------------
    def _neg_log_marginal_lik(self, log_beta, R0, nugget=None):
        beta = torch.exp(log_beta)
        R = self._kernel_matrix(beta, R0, nugget=nugget)
        try:
            L = torch.linalg.cholesky(R)
        except RuntimeError:
            # Return tensor with gradients enabled
            return torch.tensor(1e6, dtype=torch.float64, device=self.device, requires_grad=True)
    
        # use differentiable solve
        LinvY = torch.linalg.solve(R, self.Y)
        LinvF = torch.linalg.solve(R, self.trend)
    
        Q, Rf = torch.linalg.qr(LinvF, mode="reduced")
        theta_hat = torch.linalg.solve(Rf, Q.T @ LinvY)
    
        resid = LinvY - LinvF @ theta_hat
        sigma2_hat = torch.sum(resid**2, dim=0) / self.n
    
        logdetR = torch.logdet(R)   # differentiable log-determinant
        loglik = -0.5 * (self.n * torch.log(sigma2_hat) + logdetR)
    
        return -torch.sum(loglik)

    def _neg_log_marginal_post_approx_ref(self, log_beta, R0, nugget=None):
        """Posterior mode with reference-approximation prior."""
        beta = torch.exp(log_beta)
        R = self._kernel_matrix(beta, R0, nugget=nugget)
        try:
            L = torch.linalg.cholesky(R)
        except RuntimeError:
            # Return tensor with gradients enabled
            return torch.tensor(1e6, dtype=torch.float64, device=self.device, requires_grad=True)

        LinvY = torch.cholesky_solve(self.Y, L)
        LinvF = torch.cholesky_solve(self.trend, L)

        Q, Rf = torch.linalg.qr(LinvF, mode="reduced")
        theta_hat = torch.linalg.solve(Rf, Q.T @ LinvY)

        resid = LinvY - LinvF @ theta_hat
        sigma2_hat = torch.sum(resid**2, dim=0) / self.n

        logdetR = 2 * torch.sum(torch.log(torch.diag(L)))
        loglik = -0.5 * (self.n * torch.log(sigma2_hat) + logdetR)

        # prior term - ensure all operations are differentiable
        CL = torch.tensor(
            [(torch.max(self.X[:, j]) - torch.min(self.X[:, j])) / self.n ** (1 / self.p)
             for j in range(self.p)],
            dtype=torch.float64, device=self.device
        )
        a_t = torch.tensor(self.a, dtype=torch.float64, device=self.device)
        b_t = torch.tensor(self.b, dtype=torch.float64, device=self.device)
        p_t = torch.tensor(self.p, dtype=torch.float64, device=self.device)

        log_prior = torch.sum(torch.log((a_t + p_t) / (p_t * CL * b_t * beta)))
        return -(torch.sum(loglik) + log_prior)

    # --------------------
    # Training
    # --------------------
    def train(self, num_initial=3, maxiter=200, nugget_est=False):
        R0 = self._pairwise_distances()
    
        # compute CL
        CL = torch.tensor(
            [(torch.max(self.X[:, j]) - torch.min(self.X[:, j])) / self.n ** (1 / self.p)
             for j in range(self.p)],
            dtype=torch.float64, device=self.device
        )
    
        # crude lower bound
        LB = []
        for j in range(self.p):
            LB.append(-torch.log(torch.tensor(0.1, dtype=torch.float64, device=self.device)) /
                      ((torch.max(self.X[:, j]) - torch.min(self.X[:, j])) * self.p))
        LB = torch.tensor(LB, dtype=torch.float64, device=self.device)
    
        # initial guesses
        inits = []
        inits.append(torch.log(torch.full((self.p,), 50.0, dtype=torch.float64, device=self.device) * torch.exp(LB)))
        inits.append(torch.log((self.a + self.p) / (self.p * CL * self.b) / 2))
        for _ in range(num_initial - 2):
            rand_beta = (10 ** 3) * torch.rand(self.p, dtype=torch.float64, device=self.device) / CL
            inits.append(torch.log(rand_beta))
    
        if nugget_est:
            for i in range(len(inits)):
                inits[i] = torch.cat([inits[i], torch.tensor([-9.0], dtype=torch.float64, device=self.device)])
    
        # objective
        def objective(log_params):
            if nugget_est:
                log_beta, log_eta = log_params[:-1], log_params[-1]
                nugget_val = torch.exp(log_eta)
            else:
                log_beta = log_params
                nugget_val = self.nugget  # plain float, not a tensor
    
            if self.method == "post_mode" and self.prior_choice == "ref_approx":
                return self._neg_log_marginal_post_approx_ref(log_beta, R0, nugget=nugget_val)
            else:
                return self._neg_log_marginal_lik(log_beta, R0, nugget=nugget_val)
    
        # run optimization
        best_val, best_params = float("inf"), None
        for i, ini in enumerate(inits):
            # Make ini a parameter with grad
            log_params = torch.nn.Parameter(ini.clone().to(self.device))
    
            optimizer = torch.optim.LBFGS([log_params], max_iter=maxiter,
                                          history_size=200, line_search_fn="strong_wolfe")
    
            def closure():
                optimizer.zero_grad()
                loss = objective(log_params)
                
                # Debug: Check if loss requires gradients
                if not loss.requires_grad:
                    # Force gradients by creating a new tensor if needed
                    loss = loss.detach().clone().requires_grad_(True)
                
                loss.backward()
                return loss
    
            try:
                optimizer.step(closure)
                val = objective(log_params).item()
                if val < best_val:
                    best_val = val
                    best_params = log_params.detach().clone()
            except Exception as e:
                self.logger.warning(f"Init {i+1} failed: {e}")
    
        if best_params is None:
            raise RuntimeError("All optimizations failed")
    
        # extract fitted params
        if nugget_est:
            self.beta_hat = torch.exp(best_params[:-1])
            self.nugget = torch.exp(best_params[-1]).item()
        else:
            self.beta_hat = torch.exp(best_params)
    
        # recompute Cholesky
        R = self._kernel_matrix(self.beta_hat, R0, nugget=self.nugget)
        self.L = torch.linalg.cholesky(R)
    
        LinvY = torch.cholesky_solve(self.Y, self.L)
        LinvF = torch.cholesky_solve(self.trend, self.L)
        Q, Rf = torch.linalg.qr(LinvF, mode="reduced")
        self.theta_hat = torch.linalg.solve(Rf, Q.T @ LinvY)
        resid = LinvY - LinvF @ self.theta_hat
        self.sigma2_hat = torch.sum(resid**2, dim=0) / self.n
    
        self.logger.info(f"Final β: {1/self.beta_hat.cpu().numpy()}, nugget={self.nugget}")
        return self

    # --------------------
    # Prediction
    # --------------------
    def predict(self, Xnew, alpha=0.05, return_std=True, interval=True):
        Xnew = torch.as_tensor(Xnew, dtype=torch.float64, device=self.device)
        m = Xnew.shape[0]
    
        R_star = torch.ones((m, self.n), dtype=torch.float64, device=self.device)
        for j in range(self.p):
            diff = torch.abs(Xnew[:, j][:, None] - self.X[:, j])
            r = diff * self.beta_hat[j]
            sqrt5_r = torch.sqrt(torch.tensor(5.0, dtype=torch.float64)) * r
            R_star *= (1 + sqrt5_r + 5 * r**2 / 3.0) * torch.exp(-sqrt5_r)
    
        LinvY = torch.cholesky_solve(self.Y, self.L)
        LinvF = torch.cholesky_solve(self.trend, self.L)
        Q, Rf = torch.linalg.qr(LinvF, mode="reduced")
    
        Fnew = torch.ones((m, 1), dtype=torch.float64, device=self.device)
        correction = torch.cholesky_solve(self.Y - self.trend @ self.theta_hat, self.L)
        mean = Fnew @ self.theta_hat + R_star @ correction
    
        results = {"mean": mean.detach().cpu().numpy()}
    
        if return_std or interval:
            v = torch.cholesky_solve(R_star.T, self.L)
            base_var = 1.0 - torch.sum(R_star.T * v, dim=0)
            base_var = torch.clamp(base_var, min=1e-12)
    
            sd = torch.sqrt(base_var[:, None] * self.sigma2_hat[None, :])
    
            if return_std:
                results["sd"] = sd.detach().cpu().numpy()
    
            if interval:
                if self.method in ["post_mode", "mmle"]:
                    # Student-t quantile - use scipy for proper implementation
                    try:
                        from scipy.stats import t
                        z_value = t.ppf(1 - alpha / 2, self.n - 1)
                        z = torch.tensor(z_value, dtype=torch.float64, device=self.device)
                    except ImportError:
                        # Fallback: use Normal approximation if scipy not available
                        self.logger.warning("scipy not available, using Normal approximation for Student-t")
                        z = Normal(0, 1).icdf(torch.tensor(1 - alpha / 2, dtype=torch.float64))
                else:
                    z = Normal(0, 1).icdf(torch.tensor(1 - alpha / 2, dtype=torch.float64))
                
                results["lower"] = (mean - z * sd).detach().cpu().numpy()
                results["upper"] = (mean + z * sd).detach().cpu().numpy()
    
        return results
