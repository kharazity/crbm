import cirq
from cirq.circuits import Circuit
import openfermion as of
from openfermion.ops.operators import SymbolicOperator, QubitOperator, FermionOperator
from openfermion.circuits.unitary_cc import uccsd_generator
from openfermion.utils import hermitian_conjugated as hc
from openfermion.circuits import trotter_exp_to_qgates
import numpy as np
from sympy import symbols
class VQE():
    """
    class implementing VQE in cirq
    """

    def __init__(self, 
                H_op :SymbolicOperator,
                init_vals :np.array,
                qubit_map :str = "jw",
                ansatz :str = "UCCSD",
                simulate :bool = True,
                ref_state :str = "HF",
                tol :float = 1e-6,
                num_samples :int = 1e3,
                save_path :str = "."):
        """
        Args:
            H_op: openfermion operator representing our hamiltonian, must be hermitian
            tol: float that sets the tolerance for distance from ground state
            num_samples: int setting the number of samples to take from an experiment. Number of experimental repititions.
        Returns:
            VQE instance
        """
        assert(hc(H_op) == H_op)
        self.num_qubits = of.utils.count_qubits(H_op)
        self.H_op = H_op
        self.ansatz = ansatz

    def get_qubit_op(self) -> QubitOperator:
        if(self.qubit_map == "jw"):
            H_fop = of.reorder(H_fop, of.up_then_down, num_modes = self.num_qubits)
            return of.transforms.jordan_wigner(H_fop)

        
    def ansatz(self, params :np.array) -> Circuit:
        """
        
        """
        h_qop = self.get_qubit_op()
        circuit = cirq.Circuit()
        #initialize symbolic variables for parameterized circuit
        _vars = ['x{i}'.format(i = i) for i in range(len(params))]
        sym_str = ""
        for x in _vars:
            sym_str += x
        sym_vars = symbols(sym_str)
        #iterate through h_qop operators
        op_groups = h_qop

    def UCC_factors(self, qop :QubitOperator) -> Circuit:
        """
        Args: 
            qop: qubit operator term we wish to exponentiate
        Returns:
            Compiled circuit for exp(i*qop)
        """
        trotter_exp_to_qgates()



    def energy(self, params :np.array) -> float:
        state = ansatz(params)

        return 0.0

    def measure(self) -> np.ndarray:
        return [[]]


