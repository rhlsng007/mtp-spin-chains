"""
Environment check for the MTP thesis project.

Run from the project root:

    python scripts/check_env.py

Each test checks a numerical result against a known analytical answer,
so a PASS means the tool is not only installed but working correctly.
"""

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

RESULTS = []


def record(name, status, detail=""):
    RESULTS.append((name, status, detail))
    symbol = {"PASS": "✔", "FAIL": "✘", "SKIP": "–"}[status]
    print(f"  {symbol} {status:4s}  {name}" + (f"  ({detail})" if detail else ""))


def run(name, fn, required=True):
    try:
        detail = fn() or ""
        record(name, "PASS", detail)
    except ImportError as e:
        record(name, "FAIL" if required else "SKIP", f"not installed: {e.name}")
    except Exception as e:
        record(name, "FAIL", f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 1. Core numerics
# ---------------------------------------------------------------------------
def test_scipy_sparse_eigensolver():
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh

    rng = np.random.default_rng(0)
    A = sp.random(200, 200, density=0.05, random_state=0)
    A = (A + A.T).tocsr()
    e_sparse = eigsh(A, k=1, which="SA")[0][0]
    e_dense = np.linalg.eigvalsh(A.toarray())[0]
    assert abs(e_sparse - e_dense) < 1e-8, f"mismatch {e_sparse} vs {e_dense}"
    return f"lowest eigenvalue agrees to {abs(e_sparse - e_dense):.1e}"


# ---------------------------------------------------------------------------
# 2. PennyLane: noiseless state-vector simulation
# ---------------------------------------------------------------------------
def test_pennylane_statevector():
    import pennylane as qml

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def bell():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)), qml.expval(qml.PauliX(0) @ qml.PauliX(1))

    zz, xx = bell()
    assert np.isclose(zz, 1.0) and np.isclose(xx, 1.0), f"<ZZ>={zz}, <XX>={xx}"
    return f"Bell state <ZZ>={float(zz):.3f}, <XX>={float(xx):.3f}"


# ---------------------------------------------------------------------------
# 3. PennyLane: noisy density-matrix simulation
#    For a Bell state with depolarizing noise of strength p on one qubit,
#    the exact result is <ZZ> = 1 - 4p/3.
# ---------------------------------------------------------------------------
def test_pennylane_mixed():
    import pennylane as qml

    p = 0.1
    dev = qml.device("default.mixed", wires=2)

    @qml.qnode(dev)
    def noisy_bell():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.DepolarizingChannel(p, wires=0)
        return qml.density_matrix(wires=[0, 1])

    rho = np.asarray(noisy_bell())
    ZZ = np.kron(np.diag([1, -1]), np.diag([1, -1]))
    zz = np.real(np.trace(rho @ ZZ))
    purity = np.real(np.trace(rho @ rho))
    assert np.isclose(zz, 1 - 4 * p / 3), f"<ZZ>={zz}, expected {1 - 4*p/3}"
    assert purity < 1.0
    return f"<ZZ>={zz:.4f} (exact {1 - 4*p/3:.4f}), purity={purity:.4f}"


# ---------------------------------------------------------------------------
# 4. PennyLane: automatic differentiation
#    For RX(θ) on |0>, <Z> = cos θ, so the gradient is −sin θ.
# ---------------------------------------------------------------------------
def test_pennylane_gradient():
    import pennylane as qml
    from pennylane import numpy as pnp

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(theta):
        qml.RX(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    theta = pnp.array(0.7, requires_grad=True)
    grad = qml.grad(circuit)(theta)
    assert np.isclose(grad, -np.sin(0.7)), f"grad={grad}, expected {-np.sin(0.7)}"
    return f"parameter-shift gradient = {float(grad):.4f} (exact {-np.sin(0.7):.4f})"


# ---------------------------------------------------------------------------
# 5. PennyLane + PyTorch interface (needed for the QCNN in Phase 7)
# ---------------------------------------------------------------------------
def test_pennylane_torch():
    import pennylane as qml
    import torch

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev, interface="torch")
    def circuit(theta):
        qml.RX(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    theta = torch.tensor(0.7, requires_grad=True)
    circuit(theta).backward()
    assert np.isclose(theta.grad.item(), -np.sin(0.7))
    return f"torch gradient = {theta.grad.item():.4f}"


# ---------------------------------------------------------------------------
# 6. PyTorch and scikit-learn
# ---------------------------------------------------------------------------
def test_torch():
    import torch
    x = torch.randn(100, 4)
    y = torch.nn.Linear(4, 2)(x)
    assert y.shape == (100, 2)
    device = "CUDA available" if torch.cuda.is_available() else "CPU only"
    return f"torch {torch.__version__}, {device}"


def test_sklearn():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(-2, 1, (100, 2)), rng.normal(2, 1, (100, 2))])
    y = np.array([0] * 100 + [1] * 100)
    acc = LogisticRegression().fit(X, y).score(X, y)
    assert acc > 0.95, f"accuracy {acc}"
    return f"toy classification accuracy = {acc:.2f}"


# ---------------------------------------------------------------------------
# 7. Project utilities
# ---------------------------------------------------------------------------
def test_config():
    from config import CFG
    h = CFG.tfim.h_grid()
    assert np.isclose(h[0], 0.1) and np.isclose(h[-1], 2.0) and len(h) == 39
    assert CFG.noise.strengths()[0] == 0.0
    return f"{len(h)} field values, {len(CFG.noise.strengths())} noise strengths"


def test_io_roundtrip():
    from config import RESULTS_DIR
    from utils.io import save_results, load_results

    arrays = {"h": np.linspace(0, 2, 5), "energy": -np.arange(5.0)}
    save_results("roundtrip", arrays, params={"N": 4}, subdir="_env_check")
    loaded, meta = load_results("roundtrip", subdir="_env_check")
    shutil.rmtree(RESULTS_DIR / "_env_check")
    assert np.allclose(loaded["energy"], arrays["energy"])
    assert meta["params"]["N"] == 4
    return f"git commit recorded as '{meta['git_commit']}'"


def test_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from config import FIGURES_DIR
    from utils.plotting import set_thesis_style, save_figure

    set_thesis_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label=r"$S_{\mathrm{vN}}$")
    ax.legend()
    save_figure(fig, "style_test", subdir="_env_check")
    plt.close(fig)
    ok = (FIGURES_DIR / "_env_check" / "style_test.pdf").exists()
    shutil.rmtree(FIGURES_DIR / "_env_check")
    assert ok
    return "PDF and PNG export working"


def test_seeding():
    from utils.seeding import set_seed
    a = set_seed(123).normal(size=3)
    b = set_seed(123).normal(size=3)
    assert np.allclose(a, b)
    return "identical draws for identical seeds"


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\nPython {sys.version.split()[0]}  |  numpy {np.__version__}\n")

    print("Core numerics")
    run("SciPy sparse eigensolver", test_scipy_sparse_eigensolver)

    print("\nQuantum simulation (PennyLane)")
    run("State-vector simulation", test_pennylane_statevector)
    run("Density-matrix simulation with noise", test_pennylane_mixed)
    run("Parameter-shift gradients", test_pennylane_gradient)
    run("PennyLane-PyTorch interface", test_pennylane_torch)

    print("\nMachine learning")
    run("PyTorch", test_torch)
    run("scikit-learn", test_sklearn)

    print("\nProject utilities")
    run("Configuration grids", test_config)
    run("Result save/load round trip", test_io_roundtrip)
    run("Thesis plotting style", test_plotting)
    run("Reproducible seeding", test_seeding)

    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    print(f"\n{n_pass} passed, {n_fail} failed, {len(RESULTS) - n_pass - n_fail} skipped")
    if n_fail:
        print("Fix the failures above before starting Phase 1.")
        sys.exit(1)
    print("Environment ready. You can start Phase 1.")
