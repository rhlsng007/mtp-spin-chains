"""
Noiseless VQE (Phase 2).

    ansatz.py        HVA (T2.1) and HEA (T2.2) with exact adjoint gradients
    statevector.py   NumPy state-vector kernels (PennyLane gate conventions)
    optimize.py      L-BFGS-B / Adam, restarts (T2.4), warm-started sweeps (T2.5)
    gradients.py     generalized parameter-shift rule, gradient variance (T2.7)
    metrics.py       energy errors and fidelities against ED (T2.6)
    pennylane_ref.py identical circuits as PennyLane QNodes (validation, Phase 4)
"""
