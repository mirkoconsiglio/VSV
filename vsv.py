import os
import time
import json
from functools import reduce

import numpy as np
from qulacs import QuantumState, QuantumCircuit, ParametricQuantumCircuit, DensityMatrix
from qulacs.state import permutate_qubit
from scipy.optimize import *
from scipy.io import mmwrite


class BreakOut(Exception):
	pass


class VSV(object):
	def __init__(self, test_state, num_states=None, directory=None, css=None):
		# Set up input parameters
		self.test_state = test_state
		self.test_state_purity = None
		self.dimension = len(self.test_state)
		self.nqubits = int(np.log2(self.dimension))
		self.num_states = num_states if num_states else self.dimension
		self.ndim = self.test_state.ndim
		if self.ndim == 1:
			self.state = QuantumState(2 * self.nqubits)
			zero_state = np.zeros(self.dimension)
			zero_state[0] = 1
		elif self.ndim == 2:
			self.state = DensityMatrix(2 * self.nqubits)
			zero_state = np.zeros((self.dimension, self.dimension))
			zero_state[0, 0] = 1
		else:
			raise RuntimeError(f'Test state must either be a vector or matrix')
		self.single_test_state = np.kron(zero_state, self.test_state)
		self.double_test_state = np.kron(self.test_state, self.test_state)
		self.directory = directory if directory is not None else 'temp'
		self.css = css

		# Set up variational circuits
		self.dpc = self.double_params_circuit()
		self.spc = self.single_params_circuit()
		self.npc = self.no_params_circuit()
		self.permutation = [i for j in zip(range(self.nqubits), range(self.nqubits, 2 * self.nqubits)) for i in j]
		self.operator = reduce(np.kron, [[1, 1, 1, -1]] * self.nqubits)

		# Minimizer stuff
		self.min_kwargs = None
		self.params = None
		self.upper_params = None
		self.lower_params = None
		self.lower_param_bounds = [(0, 1)] * self.num_states
		self.upper_param_bounds = [(-2 * np.pi, 2 * np.pi)] * 2 * self.nqubits * self.num_states
		self.param_bounds = np.concatenate((self.lower_param_bounds, self.upper_param_bounds))
		self.lower_constraints = LinearConstraint(np.ones(self.num_states), 1, 1)
		self.constraints = LinearConstraint(
			np.concatenate((np.ones(self.num_states), np.zeros(2 * self.nqubits * self.num_states))), 1, 1)
		self.purities = None
		self.overlaps = None
		self.purity = None
		self.overlap = None

		# quantities
		self.D_list = None
		self.purity_list = None
		self.overlap_list = None
		self.fev_D_list = None
		self.fev_purity_list = None
		self.fev_overlap_list = None
		self.calculated_css = None

		# progress parameters
		self.cost = None
		self.iter = None
		self.fev = None
		self.calls = None
		self.elapsed_time = None
		self.start_time = None
		self.max_time = None
		self.max_fev = None
		self.shots = None

	def run(self, min_kwargs=None, params=None, max_calls=None, max_fevs=None, max_time=None, shots=None):
		os.makedirs(f'{self.directory}', exist_ok=True)
		# Generate parameters
		self.params = params if params is not None else np.concatenate((self.generate_random_lower_params(),
		                                                                self.generate_random_upper_params()))
		self.lower_params = self.params[:self.num_states]
		self.upper_params = self.params[self.num_states:]
		# Set break points
		self.iter = 0
		self.fev = 0
		self.calls = 0
		self.elapsed_time = 0
		if max_calls:
			self.max_fev = np.floor(2 * max_calls / (self.num_states * (self.num_states + 1)))
		elif max_fevs:
			self.max_fev = max_fevs
		else:
			self.max_fev = None
		self.max_time = max_time
		self.shots = shots
		# Create empty lists
		self.D_list = []
		self.purity_list = []
		self.overlap_list = []
		self.fev_D_list = []
		self.fev_purity_list = []
		self.fev_overlap_list = []
		# Set up minimizer kwargs
		self.min_kwargs = min_kwargs.copy() if min_kwargs is not None else dict()
		with open(f'{self.directory}/min_kwargs.json', 'w') as f:
			json.dump(self.min_kwargs, f)
		self.min_kwargs.update(func=self.upper_cost_function, x0=self.upper_params, bounds=self.upper_param_bounds,
		                       callback=self.callback)
		# Save initial calculated_css
		p = self.lower_params
		theta, phi = self.upper_params_unpacker(self.upper_params)
		self.calculated_css = self.css_density_matrix(p, theta, phi)
		mmwrite(f'{self.directory}/initial_trial_state.mtx', self.calculated_css,
		        field='complex', precision=10, symmetry='hermitian')
		# Calculate test state purity
		self.test_state_purity = self.calculate_test_state_purity()
		# Start minimization
		self.start_time = time.perf_counter()
		result = self.min_D()
		# Save last calculated_css
		p = self.lower_params
		theta, phi = self.upper_params_unpacker(self.upper_params)
		self.calculated_css = self.css_density_matrix(p, theta, phi)
		mmwrite(f'{self.directory}/calculated_calculated_css.mtx', self.calculated_css,
		        field='complex', precision=10, symmetry='hermitian')
		# Save data
		np.savetxt(f'{self.directory}/fev_D_list.txt', self.fev_D_list)
		np.savetxt(f'{self.directory}/fev_purity_list.txt', self.fev_purity_list)
		np.savetxt(f'{self.directory}/fev_overlap_list.txt', self.fev_overlap_list)
		np.savetxt(f'{self.directory}/D_list.txt', self.D_list)
		np.savetxt(f'{self.directory}/purity_list.txt', self.purity_list)
		np.savetxt(f'{self.directory}/overlap_list.txt', self.overlap_list)
		np.savetxt(f'{self.directory}/params.txt', self.params)
		np.savetxt(f'{self.directory}/metadata.txt', [self.iter, self.fev])
		# Return result
		return result

	def generate_random_upper_params(self):
		return np.random.uniform(-2 * np.pi, 2 * np.pi, 2 * self.nqubits * self.num_states)

	def generate_random_lower_params(self):
		p = np.random.uniform(0, 1, self.num_states)
		return p / np.sum(p)

	def params_unpacker(self, params):
		p = params[:self.num_states]
		theta, phi = self.upper_params_unpacker(params[self.num_states:])
		return p, theta, phi

	def upper_params_unpacker(self, upper_params):
		return upper_params[:self.nqubits * self.num_states], upper_params[self.nqubits * self.num_states:]

	def updater(self, p, theta, phi):
		self.fev += 1
		# Update params
		self.lower_params = p
		self.upper_params = np.concatenate((theta, phi))
		self.params = np.concatenate((self.lower_params, self.upper_params))
		# Check break conditions
		self.elapsed_time = time.perf_counter() - self.start_time
		if self.max_fev and self.fev >= self.max_fev:
			raise BreakOut('Maximum number of function evaluations reached')
		if self.max_time and self.elapsed_time >= self.max_time:
			raise BreakOut('Maximum time limit reached')

		# Update data
		self.fev_D_list.append(self.cost)
		self.fev_purity_list.append(self.purity)
		self.fev_overlap_list.append(self.overlap)

	def callback(self, *args):
		self.iter += 1
		# Update data
		self.D_list.append(self.cost)
		self.purity_list.append(self.purity)
		self.overlap_list.append(self.overlap)
		self.calculated_css = self.css_density_matrix(*self.params_unpacker(self.params))

		self.elapsed_time = time.perf_counter() - self.start_time
		print(f"| {self.iter} | {self.fev} | {self.calls} | {self.D_list[-1]:.10f} | {self.elapsed_time:.2f} |")

	def double_params_circuit(self):
		qc = ParametricQuantumCircuit(2 * self.nqubits)
		# Parametric gates
		for i in range(2 * self.nqubits):
			qc.add_parametric_RY_gate(i, 0)
			qc.add_parametric_RZ_gate(i, 0)
		# Destructive SWAP test
		for i in range(self.nqubits):
			qc.add_CNOT_gate(i, i + self.nqubits)
			qc.add_H_gate(i)

		return qc

	def single_params_circuit(self):
		qc = ParametricQuantumCircuit(2 * self.nqubits)
		# Parametric gates
		for i in range(self.nqubits, 2 * self.nqubits):
			qc.add_parametric_RY_gate(i, 0)
			qc.add_parametric_RZ_gate(i, 0)
		# Destructive SWAP test
		for i in range(self.nqubits):
			qc.add_CNOT_gate(i, i + self.nqubits)
			qc.add_H_gate(i)

		return qc

	def no_params_circuit(self):
		qc = QuantumCircuit(2 * self.nqubits)
		# Destructive SWAP test
		for i in range(self.nqubits):
			qc.add_CNOT_gate(i, i + self.nqubits)
			qc.add_H_gate(i)

		return qc

	def min_D(self):
		try:
			# Print headers
			print("| Iterations | Function Evaluations | Calls | HSD | Elapsed Time |")
			# Minimize
			result = dual_annealing(**self.min_kwargs)
		# Break point reached
		except BreakOut:
			return None

		return result

	def upper_cost_function(self, upper_params):
		# Get theta and phi
		theta, phi = self.upper_params_unpacker(upper_params)
		# Calculate purity and overlap lists
		self.purities = self.calculate_purities(theta, phi)
		self.overlaps = self.calculate_overlaps(theta, phi)
		# Minimize to find p
		result = minimize(self.lower_cost_function, x0=self.lower_params, bounds=self.lower_param_bounds,
		                  constraints=self.lower_constraints)
		# Get results
		p = result.x
		self.cost = result.fun
		# Update run
		self.updater(p, theta, phi)
		# Return cost
		return self.cost

	def lower_cost_function(self, p):
		self.purity, self.overlap = self.cost_function_helper(p)
		return self.test_state_purity + self.purity - 2 * self.overlap

	def cost_function_helper(self, p):
		purity = 0
		overlap = 0
		for i in range(self.num_states):
			purity += p[i] ** 2  # purity of same decomposed pure states
			overlap += p[i] * self.overlaps[i]  # overlap
			for j in range(i + 1, self.num_states):
				# purity of different decomposed pure states
				purity += 2 * p[i] * p[j] * self.purities[i][j - i - 1]

		return purity, overlap

	def cost_function(self, params):
		# Get theta and phi
		p, theta, phi = self.params_unpacker(params)
		# Calculate purity and overlap lists
		self.purities = self.calculate_purities(theta, phi)
		self.overlaps = self.calculate_overlaps(theta, phi)
		self.purity, self.overlap = self.cost_function_helper(p)
		self.cost = self.test_state_purity + self.purity - 2 * self.overlap
		# Update run
		self.updater(p, theta, phi)
		# Return cost
		return self.cost

	def calculate_purities(self, theta, phi):
		purities = []
		for i in range(self.num_states - 1):
			purity_list = []
			k = i * self.nqubits
			l = (i + 1) * self.nqubits
			for j in range(i + 1, self.num_states):
				m = j * self.nqubits
				n = (j + 1) * self.nqubits
				purity = self.swap_test(theta[k:l], phi[k:l], theta[m:n], phi[m:n])
				purity_list.append(purity)
			purities.append(purity_list)

		return purities

	def calculate_overlaps(self, theta, phi):
		overlaps = []
		for i in range(self.num_states):
			j = i * self.nqubits
			k = (i + 1) * self.nqubits
			overlap = self.swap_test(None, None, theta[j:k], phi[j:k])
			overlaps.append(overlap)

		return overlaps

	def calculate_test_state_purity(self):
		return self.swap_test(None, None, None, None)

	def swap_test(self, theta1, phi1, theta2, phi2):
		self.calls += 1
		if theta1 is not None and theta2 is not None:
			for i, [t, p] in enumerate(zip(theta1, phi1)):
				j = 2 * i
				self.dpc.set_parameter(j, t)
				self.dpc.set_parameter(j + 1, p)
			for i, [t, p] in enumerate(zip(theta2, phi2)):
				j = 2 * (i + self.nqubits)
				self.dpc.set_parameter(j, t)
				self.dpc.set_parameter(j + 1, p)
			self.state.set_zero_state()
			self.dpc.update_quantum_state(self.state)
		elif theta2 is not None:
			for i, [t, p] in enumerate(zip(theta2, phi2)):
				j = 2 * i
				self.spc.set_parameter(j, t)
				self.spc.set_parameter(j + 1, p)
			self.state.load(self.single_test_state)
			self.spc.update_quantum_state(self.state)
		else:
			self.state.load(self.double_test_state)
			self.npc.update_quantum_state(self.state)
		# Calculate overlap
		temp_state = permutate_qubit(self.state, self.permutation)
		if self.shots:
			overlap = np.sum(self.operator[i] for i in temp_state.sampling(self.shots)) / self.shots
		else:
			if self.ndim == 1:
				s = temp_state.get_vector()
			else:
				s = np.sqrt(temp_state.get_matrix().diagonal())
			overlap = np.sum(i * np.abs(j) ** 2 for i, j in zip(self.operator, s))
		# Return overlap
		return overlap

	def css_density_matrix(self, p, theta, phi):
		density_matrix = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
		for i in range(self.num_states):
			j = i * self.nqubits
			k = (i + 1) * self.nqubits
			density_matrix += p[i] * self.prod_state(theta[j:k], phi[j:k])

		return density_matrix

	def prod_state(self, theta, phi):
		qc = QuantumCircuit(self.nqubits)
		for i, [t, p] in enumerate(zip(theta, phi)):
			qc.add_RY_gate(i, t)
			qc.add_RZ_gate(i, p)
		state = QuantumState(self.nqubits)
		qc.update_quantum_state(state)
		statevector = state.get_vector()
		return np.outer(statevector, statevector.conj())
