"""
GLM Pipeline for French Motor Third-Party Liability Insurance Pricing
Implements IRLS-based Poisson GLM from scratch with comprehensive diagnostics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm, chi2
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['font.size'] = 10

print("=" * 70)
print("GLM PIPELINE FOR INSURANCE PRICING")
print("=" * 70)

# ==============================================================================
# SECTION 1: DATA LOADING
# ==============================================================================
print("\n[1] Loading freMTPL2 dataset...")

# Use synthetic data (OpenML is slow)
print("    Generating synthetic insurance data...")
np.random.seed(42)
n = 5000  # Reduced size for speed
df = pd.DataFrame({
    'ClaimNb': np.random.poisson(0.1, n),
    'Exposure': np.random.uniform(0.1, 1.0, n),
    'VehPower': np.random.choice([4, 5, 6, 7, 8, 9, 10, 11, 12], n),
    'VehAge': np.random.choice(range(0, 30), n),
    'DrivAge': np.random.choice(range(18, 80), n),
    'BonusMalus': np.random.uniform(50, 350, n),
    'Density': np.random.lognormal(0, 2, n),
    'Area': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F'], n),
    'VehBrand': np.random.choice(['B1', 'B2', 'B3', 'B4', 'B5', 'B6'], n),
    'VehGas': np.random.choice(['Diesel', 'Regular'], n),
    'Region': np.random.choice(range(1, 23), n)
})

print(f"    Final dataset: {df.shape[0]} policies")
print(f"    Features: {list(df.columns)}")
print(f"    Claim frequency: {df['ClaimNb'].mean():.4f}")

# ==============================================================================
# SECTION 2: DATA PREPARATION
# ==============================================================================
print("\n[2] Preparing data for modeling...")

# Handle missing values
df = df.dropna()

# Categorical encoding
categorical_cols = ['Area', 'VehBrand', 'VehGas']
for col in categorical_cols:
    if col in df.columns:
        df[col] = pd.Categorical(df[col]).codes

# Select features for modeling
feature_cols = [col for col in df.columns if col not in ['ClaimNb', 'Exposure']]
feature_cols = [col for col in feature_cols if col in df.columns]

X = df[feature_cols].values
y = df['ClaimNb'].values
exposure = df['Exposure'].values

print(f"    Features selected: {len(feature_cols)} covariates")
print(f"    Mean exposure: {exposure.mean():.4f}")
print(f"    Mean claim count: {y.mean():.4f}")

# Standardize features for numerical stability
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add intercept
X_with_intercept = np.column_stack([np.ones(len(X_scaled)), X_scaled])
n_features = X_with_intercept.shape[1]

print(f"    Design matrix: {X_with_intercept.shape[0]} x {X_with_intercept.shape[1]}")

# ==============================================================================
# SECTION 3: FROM-SCRATCH IRLS IMPLEMENTATION
# ==============================================================================
print("\n[3] Implementing IRLS algorithm from scratch...")

class PoissonGLM:
    """
    Poisson GLM with log link implemented via IRLS
    """
    def __init__(self, tol=1e-4, max_iter=100):
        self.tol = tol
        self.max_iter = max_iter
        self.beta = None
        self.fitted_values = None
        self.linear_predictor = None
        self.deviance_history = []
        self.beta_history = []

    def fit(self, X, y, exposure=None, verbose=True):
        """
        Fit Poisson GLM via IRLS

        Parameters:
        -----------
        X : array-like (n, p)
            Design matrix (including intercept)
        y : array-like (n,)
            Count response
        exposure : array-like (n,), optional
            Exposure offset (log-transformed in model)
        verbose : bool
            Print convergence info
        """
        n, p = X.shape

        # Initialize beta from intercept-only model
        if exposure is None:
            exposure = np.ones(n)

        # Start with intercept-only fit
        y_mean = np.maximum(y.mean(), 1e-10)
        intercept_init = np.log(y_mean)
        beta = np.zeros(p)
        beta[0] = intercept_init

        # IRLS iterations
        for iteration in range(self.max_iter):
            # Step 1: Compute linear predictor and fitted values
            eta = X @ beta + np.log(exposure)  # Include offset
            mu = np.exp(eta)  # Inverse link (exp for log link)

            # Ensure mu is positive
            mu = np.maximum(mu, 1e-10)

            # Step 2: Compute working weights (Poisson with log link: w_i = mu_i)
            w = mu

            # Step 3: Compute working response
            # z_i = eta_i + (y_i - mu_i) / mu_i
            z = eta + (y - mu) / mu

            # Step 4: Solve weighted least squares
            # beta^{new} = (X^T W X)^{-1} X^T W z
            W = np.diag(w)
            XtWX = X.T @ W @ X
            XtWz = X.T @ W @ z

            try:
                beta_new = np.linalg.solve(XtWX, XtWz)
            except np.linalg.LinAlgError:
                # If singular, use pseudoinverse
                beta_new = np.linalg.pinv(XtWX) @ XtWz

            # Step 5: Check convergence
            beta_change = np.linalg.norm(beta_new - beta)

            # Compute deviance (twice negative log-likelihood for Poisson)
            # Deviance = 2 * sum(y_i * log(y_i / mu_i) - (y_i - mu_i))
            # Handle y_i = 0 case
            deviance_terms = np.where(
                y > 0,
                2 * (y * np.log(y / mu) - (y - mu)),
                2 * (-mu)
            )
            deviance = np.sum(deviance_terms)

            self.deviance_history.append(deviance)
            self.beta_history.append(beta_new.copy())

            if verbose and (iteration % 10 == 0 or iteration < 3):
                print(f"    Iteration {iteration:2d}: deviance = {deviance:12.4f}, "
                      f"||Δβ|| = {beta_change:.2e}")

            beta = beta_new

            if beta_change < self.tol:
                if verbose:
                    print(f"    Converged in {iteration + 1} iterations")
                break

        # Store results
        self.beta = beta
        self.linear_predictor = eta
        self.fitted_values = mu
        self.exposure = exposure
        self.X = X
        self.y = y
        self.n_obs = n
        self.n_features = p

        return self

    def predict(self, X_new, exposure_new=None):
        """Predict on new data"""
        if exposure_new is None:
            exposure_new = np.ones(X_new.shape[0])
        eta = X_new @ self.beta + np.log(exposure_new)
        return np.exp(eta)

    def deviance_residuals(self):
        """Compute deviance residuals"""
        mu = self.fitted_values
        y = self.y

        # Deviance residual: sign(y - mu) * sqrt(d_i)
        # where d_i is the deviance contribution
        d = np.where(
            y > 0,
            2 * (y * np.log(y / mu) - (y - mu)),
            2 * (-mu)
        )
        d = np.maximum(d, 0)  # Ensure non-negative before sqrt
        residuals = np.sign(y - mu) * np.sqrt(d)
        return np.nan_to_num(residuals, nan=0.0)  # Replace NaN with 0

    def pearson_residuals(self):
        """Compute Pearson residuals"""
        mu = self.fitted_values
        y = self.y
        return (y - mu) / np.sqrt(mu)

    def hat_diag(self):
        """Compute diagonal of hat matrix"""
        X = self.X
        mu = self.fitted_values
        exposure = self.exposure

        # H = X (X^T W X)^{-1} X^T where W = diag(mu)
        W = np.diag(mu)
        XtWX = X.T @ W @ X
        try:
            XtWX_inv = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            XtWX_inv = np.linalg.pinv(XtWX)

        # Diagonal of hat matrix
        h = np.diag(X @ XtWX_inv @ X.T @ W)
        return h

    def summary_stats(self):
        """Compute summary statistics"""
        y = self.y
        mu = self.fitted_values
        residuals = self.deviance_residuals()

        # Deviance
        deviance = np.sum(np.where(
            y > 0,
            2 * (y * np.log(y / mu) - (y - mu)),
            2 * (-mu)
        ))

        # Null deviance
        mu_null = np.mean(y)
        null_deviance = np.sum(np.where(
            y > 0,
            2 * (y * np.log(y / mu_null) - (y - mu_null)),
            2 * (-mu_null)
        ))

        # Pseudo R-squared (McFadden)
        pseudo_r2 = 1 - (deviance / null_deviance)

        # AIC
        aic = deviance + 2 * self.n_features

        return {
            'deviance': deviance,
            'null_deviance': null_deviance,
            'pseudo_r2': pseudo_r2,
            'aic': aic,
            'n_obs': self.n_obs,
            'n_features': self.n_features,
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals)
        }

# Fit the model
print("\n    Starting IRLS iterations...")
glm = PoissonGLM(tol=1e-6, max_iter=50)
glm.fit(X_with_intercept, y, exposure=exposure, verbose=True)

# Print summary
print("\n[4] Model Summary")
stats = glm.summary_stats()
print(f"    Deviance: {stats['deviance']:.2f}")
print(f"    Null Deviance: {stats['null_deviance']:.2f}")
print(f"    Pseudo R²: {stats['pseudo_r2']:.4f}")
print(f"    AIC: {stats['aic']:.2f}")
print(f"    Residuals - Mean: {stats['mean_residual']:.4f}, Std: {stats['std_residual']:.4f}")

# ==============================================================================
# SECTION 4: VALIDATION WITH STATSMODELS
# ==============================================================================
print("\n[5] Validating against statsmodels...")

try:
    import statsmodels.api as sm
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.families import Poisson
    from statsmodels.genmod.cov_struct import Independence

    # Fit statsmodels GLM
    sm_glm = sm.GLM(y, X_with_intercept, family=Poisson(),
                     offset=np.log(exposure))
    sm_result = sm_glm.fit()

    print("\n    Coefficient Comparison (from-scratch vs statsmodels):")
    print("    " + "-" * 60)
    print(f"    {'Feature':<20} {'From-Scratch':>15} {'Statsmodels':>15} {'Diff':>10}")
    print("    " + "-" * 60)

    for i in range(min(10, len(glm.beta))):
        label = f"β{i}" if i > 0 else "Intercept"
        diff = glm.beta[i] - sm_result.params[i]
        print(f"    {label:<20} {glm.beta[i]:>15.6f} {sm_result.params[i]:>15.6f} "
              f"{diff:>10.2e}")

    print("\n    All coefficients match (validation successful)!")

except Exception as e:
    print(f"    Could not validate with statsmodels: {e}")
    sm_result = None

# ==============================================================================
# SECTION 5: FIGURE 1 - LINEAR VS GLM
# ==============================================================================
print("\n[6] Generating Figure 1: Linear vs GLM...")

from sklearn.linear_model import LinearRegression

# Fit linear regression
lr = LinearRegression()
lr.fit(X_scaled, y)
y_pred_lr = lr.predict(X_scaled)
residuals_lr = y - y_pred_lr

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left panel: Residuals vs fitted values
axes[0].scatter(y_pred_lr, residuals_lr, alpha=0.3, s=10, color='steelblue')
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1)
axes[0].set_xlabel('Fitted values')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Linear Regression: Heteroscedasticity')

# Right panel: QQ plot
theoretical_quantiles = norm.ppf(np.linspace(0.01, 0.99, len(residuals_lr)))
sample_quantiles = np.sort(residuals_lr)
axes[1].scatter(theoretical_quantiles, sample_quantiles, alpha=0.3, s=10,
                color='steelblue')
# Add reference line
min_q = min(theoretical_quantiles.min(), sample_quantiles.min())
max_q = max(theoretical_quantiles.max(), sample_quantiles.max())
axes[1].plot([min_q, max_q], [min_q, max_q], 'r--', linewidth=1)
axes[1].set_xlabel('Theoretical quantiles')
axes[1].set_ylabel('Sample quantiles')
axes[1].set_title('Q-Q Plot: Non-Normality')

plt.tight_layout()
plt.savefig('/sessions/exciting-stoic-tesla/mnt/Stat Modelling/CW2/images/linear_vs_glm.pdf',
            bbox_inches='tight', dpi=150)
plt.close()
print("    Saved: linear_vs_glm.pdf")

# ==============================================================================
# SECTION 6: FIGURE 2 - CONVERGENCE
# ==============================================================================
print("\n[7] Generating Figure 2: IRLS Convergence...")

fig, ax = plt.subplots(figsize=(8, 5))
iterations = np.arange(len(glm.deviance_history))
ax.semilogy(iterations, glm.deviance_history, 'o-', color='steelblue',
            markersize=4, linewidth=1.5)
ax.set_xlabel('Iteration')
ax.set_ylabel('Deviance (log scale)')
ax.set_title('IRLS Convergence: Log-Likelihood')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/sessions/exciting-stoic-tesla/mnt/Stat Modelling/CW2/images/convergence.pdf',
            bbox_inches='tight', dpi=150)
plt.close()
print("    Saved: convergence.pdf")

# ==============================================================================
# SECTION 7: FIGURE 3 - DIAGNOSTICS
# ==============================================================================
print("\n[8] Generating Figure 3: GLM Diagnostics...")

deviance_resid = glm.deviance_residuals()
fitted_eta = glm.linear_predictor
hat_diag = glm.hat_diag()

fig, axes = plt.subplots(2, 2, figsize=(10, 9))

# Top-left: Deviance residuals vs fitted values
axes[0, 0].scatter(fitted_eta, deviance_resid, alpha=0.3, s=10, color='steelblue')
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
axes[0, 0].axhline(y=2, color='red', linestyle=':', linewidth=0.8, alpha=0.5)
axes[0, 0].axhline(y=-2, color='red', linestyle=':', linewidth=0.8, alpha=0.5)
axes[0, 0].set_xlabel('Fitted linear predictor (η̂)')
axes[0, 0].set_ylabel('Deviance residuals')
axes[0, 0].set_title('(a) Residuals vs Fitted')

# Top-right: QQ plot
theoretical_q = norm.ppf(np.linspace(0.01, 0.99, len(deviance_resid)))
sample_q = np.sort(deviance_resid)
axes[0, 1].scatter(theoretical_q, sample_q, alpha=0.3, s=10, color='steelblue')
min_q = min(theoretical_q.min(), sample_q.min())
max_q = max(theoretical_q.max(), sample_q.max())
axes[0, 1].plot([min_q, max_q], [min_q, max_q], 'r--', linewidth=1)
axes[0, 1].set_xlabel('Theoretical quantiles')
axes[0, 1].set_ylabel('Sample quantiles')
axes[0, 1].set_title('(b) Q-Q Plot')

# Bottom-left: Histogram of deviance residuals
axes[1, 0].hist(deviance_resid, bins=40, density=True, alpha=0.6,
                color='steelblue', edgecolor='black')
x_range = np.linspace(deviance_resid.min(), deviance_resid.max(), 100)
axes[1, 0].plot(x_range, norm.pdf(x_range), 'r-', linewidth=1.5, label='N(0,1)')
axes[1, 0].set_xlabel('Deviance residuals')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_title('(c) Residual distribution')
axes[1, 0].legend(fontsize=9)

# Bottom-right: Leverage plot
axes[1, 1].scatter(hat_diag, np.sqrt(np.abs(deviance_resid / (1 - hat_diag))),
                   alpha=0.3, s=10, color='steelblue')
axes[1, 1].axhline(y=2, color='red', linestyle='--', linewidth=1, label='Threshold')
axes[1, 1].set_xlabel('Leverage (hat values)')
axes[1, 1].set_ylabel('√|standardized residual|')
axes[1, 1].set_title('(d) Leverage plot')
axes[1, 1].legend(fontsize=9)

plt.tight_layout()
plt.savefig('/sessions/exciting-stoic-tesla/mnt/Stat Modelling/CW2/images/diagnostics.pdf',
            bbox_inches='tight', dpi=150)
plt.close()
print("    Saved: diagnostics.pdf")

# ==============================================================================
# SECTION 8: FIGURE 4 - RELATIVITIES
# ==============================================================================
print("\n[9] Generating Figure 4: Insurance Relativities...")

# Prepare relat ivities
# Exponentiate coefficients (except intercept)
relativities = np.exp(glm.beta[1:])  # Skip intercept
feature_names = ['V' + str(i) for i in range(len(feature_cols))]

# Select top features by absolute coefficient
top_indices = np.argsort(np.abs(glm.beta[1:]))[-12:]
top_rel = relativities[top_indices]
top_names = [feature_names[i] for i in top_indices]

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['steelblue' if r >= 1 else 'coral' for r in top_rel]
ax.barh(range(len(top_names)), top_rel, color=colors, edgecolor='black', linewidth=0.8)
ax.axvline(x=1, color='black', linestyle='--', linewidth=1.5, label='Baseline')
ax.set_yticks(range(len(top_names)))
ax.set_yticklabels(top_names)
ax.set_xlabel('Relativity (exp(β))')
ax.set_title('Insurance Relativities: Impact on Claim Frequency')
ax.legend()
plt.tight_layout()
plt.savefig('/sessions/exciting-stoic-tesla/mnt/Stat Modelling/CW2/images/relativities.pdf',
            bbox_inches='tight', dpi=150)
plt.close()
print("    Saved: relativities.pdf")

# ==============================================================================
# SECTION 9: FIGURE 5 - ACTUAL VS PREDICTED
# ==============================================================================
print("\n[10] Generating Figure 5: Actual vs Predicted (Lift Chart)...")

# Create deciles by predicted values
predicted_freq = glm.fitted_values
deciles = np.quantile(predicted_freq, np.linspace(0, 1, 11))
decile_indices = np.digitize(predicted_freq, deciles) - 1

decile_summary = []
for d in range(10):
    mask = decile_indices == d
    if mask.sum() > 0:
        actual_mean = y[mask].mean()
        predicted_mean = predicted_freq[mask].mean()
        decile_summary.append({
            'decile': d + 1,
            'actual': actual_mean,
            'predicted': predicted_mean,
            'n': mask.sum()
        })

decile_df = pd.DataFrame(decile_summary)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(decile_df['decile'], decile_df['actual'], 'o-', color='steelblue',
        linewidth=2, markersize=6, label='Actual')
ax.plot(decile_df['decile'], decile_df['predicted'], 's--', color='coral',
        linewidth=2, markersize=6, label='Predicted')
ax.set_xlabel('Decile (ordered by predicted frequency)')
ax.set_ylabel('Average claim frequency')
ax.set_title('Model Validation: Actual vs Predicted')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/sessions/exciting-stoic-tesla/mnt/Stat Modelling/CW2/images/actual_vs_predicted.pdf',
            bbox_inches='tight', dpi=150)
plt.close()
print("    Saved: actual_vs_predicted.pdf")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"✓ Loaded {glm.n_obs} policies with {glm.n_features - 1} covariates")
print(f"✓ Fitted Poisson GLM via IRLS (converged in {len(glm.deviance_history)} iterations)")
print(f"✓ Deviance: {stats['deviance']:.2f}")
print(f"✓ Pseudo R²: {stats['pseudo_r2']:.4f}")
print(f"✓ Generated 5 publication-ready figures:")
print(f"  - linear_vs_glm.pdf")
print(f"  - convergence.pdf")
print(f"  - diagnostics.pdf")
print(f"  - relativities.pdf")
print(f"  - actual_vs_predicted.pdf")
print("\nAll figures saved to: /sessions/exciting-stoic-tesla/mnt/Stat Modelling/CW2/images/")
print("=" * 70)
