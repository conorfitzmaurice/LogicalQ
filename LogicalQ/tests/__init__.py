# Shared test configuration

# Default absolute tolerance for fidelity == 1.0 comparisons on exact (non-approximated) gates.
# Matches numpy's default atol so existing `np.isclose(fidelity, 1.0)` call sites behave identically.
FIDELITY_ATOL = 1E-8

# Absolute tolerance for fidelity == 1.0 comparisons on rotation gates, which are realized via the
# Solovay-Kitaev approximation and therefore do not achieve the same numerical accuracy as exact gates.
ROTATION_FIDELITY_ATOL = 1E-2
