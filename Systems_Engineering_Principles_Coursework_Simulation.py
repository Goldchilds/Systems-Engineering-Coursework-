import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom, norm

# Parameters
n_simulations = 10000  # Number of Monte Carlo simulations
ups_failure_rate = 0.0001  # Probability of UPS failure
critical_operations = 100  # Number of critical operations dependent on UPS

# Simulate UPS Failures
np.random.seed(42)  # For reproducibility
ups_failures = np.random.binomial(critical_operations, ups_failure_rate, n_simulations)

# Convert to DataFrame
results = pd.DataFrame({
    "UPS Failures": ups_failures
})

# 1. Statistical Analysis: Quantify Uncertainty
mean_failures = results["UPS Failures"].mean()
std_failures = results["UPS Failures"].std()

# Confidence Interval (95%)
conf_interval = (mean_failures - 1.96 * std_failures, mean_failures + 1.96 * std_failures)

print(f"Mean UPS Failures: {mean_failures:.2f}")
print(f"Standard Deviation: {std_failures:.2f}")
print(f"95% Confidence Interval: {conf_interval[0]:.2f} to {conf_interval[1]:.2f}")

# Probability of no failures in a single outage
prob_no_failure = binom.pmf(0, critical_operations, ups_failure_rate)
print(f"Probability of No UPS Failures: {prob_no_failure * 100:.2f}%")

# 2. Simulation Analysis: Assess Impact
# Calculate percentage of outages affecting >5 critical operations
threshold = 1
prob_above_threshold = (results["UPS Failures"] >= threshold).mean()
print(f"Probability of >={threshold} UPS Failures: {prob_above_threshold * 100:.2f}%")

# Visualization: Distribution of UPS Failures
plt.figure(figsize=(10, 6))
plt.hist(results["UPS Failures"], bins=range(0, max(ups_failures)+1), alpha=0.7, edgecolor="black")
plt.xlabel("Number of UPS Failures")
plt.ylabel("Frequency")
plt.title("Distribution of UPS Failures")
plt.show()
