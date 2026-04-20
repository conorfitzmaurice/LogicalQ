import inspect

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate, RYGate, RZGate
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit.transpiler.passes import SolovayKitaev

# Shared test configuration
#
# Tolerances are derived from their underlying sources rather than hardcoded, so the suite
# self-calibrates to the numerical/approximation regime it is actually running in.

# Absolute tolerance for fidelity == 1.0 comparisons on exact (non-approximated) gates.
# Derived from numpy's documented default `atol` for `np.isclose`, so any drift in the
# numpy convention automatically propagates to the test suite.
FIDELITY_ATOL: float = inspect.signature(np.isclose).parameters["atol"].default


def sk_rotation_atol(
    recursion_degree: int = 1,
    depth: int = 10,
    basis_gates=None,
    thetas=None,
) -> float:
    """Empirically derive the fidelity tolerance induced by the Solovay-Kitaev approximation
    that `LogicalCircuit.rx/ry/rz` uses to discretize rotation gates.

    The bound is computed as the worst-case state infidelity between `U_theta |0>` and
    `SK(U_theta) |0>` over a sweep of angles and rotation axes, at the same
    `recursion_degree`, `depth`, and `basis_gates` the logical rotation gates use. This
    ties the tolerance directly to the parameters driving the approximation error rather
    than relying on a hardcoded constant.
    """

    if basis_gates is None:
        basis_gates = ["s", "sdg", "t", "tdg", "h", "x", "y", "z", "cz"]

    if thetas is None:
        thetas = list(np.linspace(0, 2*np.pi, 8, endpoint=False))

    sk = SolovayKitaev(recursion_degree=recursion_degree, basis_gates=basis_gates, depth=depth)

    worst_infidelity = 0.0
    for gate_cls in (RXGate, RYGate, RZGate):
        for theta in thetas:
            qc_ideal = QuantumCircuit(1)
            qc_ideal.append(gate_cls(theta), [0])

            qc_sk = sk(qc_ideal)

            sv_ideal = Statevector.from_int(0, dims=2).evolve(qc_ideal)
            sv_sk = Statevector.from_int(0, dims=2).evolve(qc_sk)

            infidelity = 1.0 - state_fidelity(sv_ideal, sv_sk, validate=False)
            worst_infidelity = max(worst_infidelity, infidelity)

    return float(worst_infidelity)
