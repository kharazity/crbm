#%%
import CRBM
from CRBM import train
#%%
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
import openfermion as of
from openfermion.ops import SymbolicOperator, FermionOperator, QubitOperator
from openfermion.hamiltonians import generate_hamiltonian
from openfermion.transforms import jordan_wigner, bravyi_kitaev, get_fermion_operator, binary_code_transform, checksum_code, reorder
from openfermion.utils import up_then_down
from openfermion.ops import BinaryCode
import numpy as np

run_scf = 1
run_mp2 = 1
run_cisd = 0
run_ccsd = 0
run_fci = 1

basis = 'sto-3g'
multiplicity = 1
bond_length = 0.7
geometry  = [('H', (0., 0., 0.)), ('H', (0., 0., bond_length))]
h2_molecule = MolecularData(geometry, basis, multiplicity, 
                        description = str(round(bond_length, 2)))
h2_molecule.load()
hamiltonian = h2_molecule.get_molecular_hamiltonian()
f_op = get_fermion_operator(hamiltonian)
up_down = binary_code_transform(reorder(f_op, up_then_down), 2*checksum_code(2, 1))

hidden_units = 5
crbm = CRBM.CRBM(up_down, hidden_units)

H_mat = of.linalg.get_sparse_operator(up_down)
psi = of.linalg.get_ground_state(H_mat)[1]

print(psi)


#Is this the main training loop?
#should perhaps be added as a method of CRBM module or as part 
#of the CRBM class methods
from tqdm.notebook import trange
res = []
x = np.logspace(2.5, 4.5, num = 8).astype(int)
x = (x//20) * 20
for n_sample in x:
    res.append([])
    for i in trange(100):
        print(i*n_sample)
        sv2 = train(crbm, psi, seed = i*n_sample)
        #expectationvalue H_mat wrt sv2
        E = of.linalg.expectation(H_mat, sv2)
        res[-1].append(E)