"""Make quantum circuits such as QAOA, SWAP QAOA or VQE."""


from qiskit import QuantumCircuit


def make_QAOA(N, params, Σ, g):
    """Make QAOA circuit.

    Parameters
    ----------
    N: int
        Number of qubits.

    params: 1d array of floats
        Gate parameters. Size of vector is 2*p, with p the number of
        QAOA layers. The first p parameters are the angles of the CZ
        gates and the others are the angles of the X gates.

    Σ: 2d array of floats
        Stock correlation matrix.

    g: 1d array of floats
        Correlation with index vector.

    Return
    ------
    qc: Qiskit Quantum Circuit
        QAOA quantum circuit.

    """
    
    # Circuit layers and parameters.
    p = params.size//2
    γ_vals = params[:p]
    β_vals = params[p:]
    
    # Prepare quantum circuit.
    qc = QuantumCircuit(N, N)

    # Initialize in eigenstates of X.
    qc.h(range(N))
    qc.barrier()

    # Make circuit layers.
    for ip in range(p):
        # Make Hamiltonian rotations.
        for i in range(N):
            for j in range(i+1, N):
                qc.cp(-2*γ_vals[ip]*Σ[i, j], i, j)

        for i in range(N):
            qc.p(-γ_vals[ip]*(Σ[i, i] - 2*g[i]), i)

        qc.barrier()

        # Make drift Hamiltonian.
        for i in range(N):
            qc.rx(-β_vals[ip], i)

        qc.barrier()

    # Measurement.
    qc.measure(range(N), range(N))

    return qc