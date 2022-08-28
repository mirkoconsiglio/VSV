import numpy as np
from scipy.linalg import sqrtm
from scipy.sparse import dok_matrix

def hsd(rho, sigma):
	rho_sigma = rho - sigma
	return np.trace(rho_sigma @ rho_sigma).real


def fidelity(rho, sigma):
	sqrt_rho = sqrtm(rho)
	return np.trace(sqrtm(sqrt_rho @ np.array(sigma) @ sqrt_rho)).real ** 2


def GHZ(nqubits):
	d = 2 ** nqubits

	test_state = np.zeros(d)
	test_state[0] = 1 / np.sqrt(2)
	test_state[-1] = 1 / np.sqrt(2)

	css = dok_matrix((d, d), dtype=np.complex128)
	x = ((d - 2) ** 2) / (4 + 4 ** nqubits - 2 ** (nqubits + 1))
	p0 = x / 2
	p1 = (1 - x) / d
	for i in range(d):
		css[i, i] = p1
	css[0, d - 1] = p1
	css[d - 1, 0] = p1
	css[0, 0] += p0
	css[d - 1, d - 1] += p0
	css = np.asarray(css.todense())

	dist = (d - 2) / (-4 + 2. ** (3 - nqubits) + 2 ** (nqubits + 1))

	return test_state, css, dist


def mixed_state_params(nqubits):
	dimension = 2 ** nqubits

	params = [1 / dimension for _ in range(dimension)]

	for i in range(dimension):
		for j in f'{i:0{nqubits}b}':
			params.append(int(j) * np.pi)

	params.extend(np.zeros(nqubits * dimension))

	return params


def dephased_GHZ_params(nqubits):
	dimension = 2 ** nqubits

	params = list(np.zeros(dimension))
	params[0] = 0.5
	params[-1] = 0.5

	theta = np.zeros(nqubits * dimension)
	for i in range(nqubits):
		theta[-i - 1] = 1
	params.extend(theta)

	params.extend(np.zeros(nqubits * dimension))

	return params


def X_MEM(nqubits, gamma=None):
	if gamma is None:
		gamma = np.random.uniform(0, 1 / 2) * np.exp(1j * np.random.uniform(0, 2 * np.pi))
	assert 0 <= np.abs(gamma) <= 1 / 2

	n = 2 ** (nqubits - 1)
	if 0 <= np.abs(gamma) <= 1 / (n + 1):
		f = 1 / (n + 1)
		g = 1 / (n + 1)
	else:
		f = np.abs(gamma)
		g = (1 - 2 * np.abs(gamma)) / (n - 1)

	test_state = np.zeros((2 ** nqubits, 2 ** nqubits), dtype=np.complex128)
	test_state[0, 0] = f
	test_state[-1, -1] = f
	test_state[0, -1] = gamma
	test_state[-1, 0] = np.conj(gamma)
	for i in range(1, n):
		test_state[i, i] = g

	return test_state


def random_X(nqubits):
	dim = 2 ** nqubits
	# Diagonal
	a = np.random.uniform(0, 1, dim)
	a /= np.sum(a)
	test_state = np.zeros((dim, dim), dtype=np.complex128)
	# Construct density matrix
	for i in range(dim >> 1):
		z = np.random.uniform(0, np.sqrt(a[i] * a[-i - 1])) * np.exp(2j * np.pi * np.random.uniform(0, 1))
		test_state[i, i] = a[i]
		test_state[-i - 1, -i - 1] = a[-i - 1]
		test_state[i, -i - 1] = z
		test_state[-i - 1, i] = np.conj(z)

	return test_state


def x_concurrence(state):
	dim = len(state)
	dim2 = int(len(state) / 2)
	c = 0
	for i in range(dim2):
		z = np.abs(state[i, dim - i - 1])
		a = np.sum([np.sqrt(state[j, j] * state[dim - j - 1, dim - j - 1]).real
		            for j in range(dim2) if j != i])
		c = np.max([c, z - a])

	return 2 * c


def random_two_qubit_state():
	A = np.matrix(np.random.uniform(0, 1, (4, 4)) + 1j * np.random.uniform(0, 1, (4, 4)))
	A = A @ A.H
	return A / np.trace(A)


def concurrence(state):
	sy2 = np.array([[0, 0, 0, -1],
	                [0, 0, 1, 0],
	                [0, 1, 0, 0],
	                [-1, 0, 0, 0]])
	rho = state @ sy2 @ state.conj() @ sy2
	eigs = np.sort(np.sqrt(np.linalg.eig(rho)[0]))[::-1].real
	return np.max([0, eigs[0] - eigs[1] - eigs[2] - eigs[3]])


def X_MEMS(gamma=np.linspace(0, 0.5, 51), nqubits=1, min_kwargs=None, max_calls=None,
           max_fevs=None, max_time=None, shots=None, adiabatic_assistance=True, directory='temp'):
	min_kwargs = min_kwargs or dict()
	params = None
	for g in np.array(gamma).flatten():
		folder = f'{nqubits}_qubits_{shots}_shots' if shots else 'statevector'

		test_state = X_MEM(nqubits, g)
		directory_ = f'{directory}/{folder}/X_MEM_gamma_{g:.2f}'
		print(f'Gamma: {g}')
		print(test_state)
		print(f'GME Concurrence: {2 * np.abs(g)}')

		G = VSV(test_state, directory=directory_)
		result = G.run(min_kwargs, params, max_calls, max_fevs, max_time, shots)

		print(result)
		print()

		if adiabatic_assistance:
			params = G.params


def GHZs(nqubits=None, min_kwargs=None, max_calls=None, max_fevs=None,
         max_time=None, shots=None, directory='temp'):
	if nqubits is None:
		nqubits = range(2, 6)
	nqubits = np.asarray(nqubits).flatten()
	min_kwargs = min_kwargs  or dict()
	for n in nqubits:
		folder = f'{shots}_shots' if shots else 'statevector'

		test_state, css, dist = GHZ(n)
		directory_ = f'{directory}/{folder}/GHZ_{n}'
		print(f'Number of qubits: {n}')
		print(test_state)
		print(dist)

		G = VSV(test_state, directory=directory_)
		G.run(min_kwargs, max_fevs, max_time, shots)

		print(f'Calculated HSD: {G.D_list[-1]}')
		print(G.calculated_css)
		print()


if __name__ == '__main__':
	shots = 1024
	directory = 'NGSA'

	if shots:
		from vsv import VSV
	else:
		from classical_vsv import VSV

	min_kwargs = dict(maxfun=1000, initial_temp=5230, no_local_search=True,
	                  restart_temp_ratio=1e-10, visit=2.62, accept=-5, last_avg=5)

	args = dict(nqubits=2, min_kwargs=min_kwargs, shots=shots, directory=directory)

	X_MEMS(**args)

	# GHZs(**args)
