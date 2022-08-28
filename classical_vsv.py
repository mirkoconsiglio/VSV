import os
import time
import json

import numpy as np
from qulacs import QuantumCircuit, QuantumState
from scipy.io import mmwrite
from scipy.optimize import *
from qiskit.algorithms.optimizers import *


class BreakOut(Exception):
	pass


class VSV(object):
	def __init__(self, test_state, num_states=None, directory=None):
		# Set up input parameters
		self.test_state = test_state
		if self.test_state.ndim == 1:
			self.test_state = np.outer(self.test_state, self.test_state.conj())
		self.test_state_purity = np.trace(self.test_state @ self.test_state).real
		self.dimension = len(self.test_state)
		self.nqubits = int(np.log2(self.dimension))
		self.num_states = num_states if num_states else self.dimension
		self.directory = directory if directory is not None else 'temp'

		# Minimizer
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
		self.purity = None
		self.overlap = None
		self.pure_states = None

		# quantities
		self.D_list = None
		self.purity_list = None
		self.overlap_list = None
		self.css = None

		# alternative quantities
		self.fev_D_list = None
		self.fev_purity_list = None
		self.fev_overlap_list = None

		# progress parameters
		self.cost = None
		self.iter = None
		self.fev = None
		self.elapsed_time = None
		self.start_time = None
		self.max_time = None
		self.max_fev = None

	def run(self, min_kwargs=None, params=None, max_fevs=None, max_time=None):
		os.makedirs(f'{self.directory}', exist_ok=True)
		# Generate parameters
		self.params = params or np.concatenate((self.generate_random_lower_params(),
		                                        self.generate_random_upper_params()))
		self.lower_params = self.params[:self.num_states]
		self.upper_params = self.params[self.num_states:]
		# Set break points
		self.iter = 0
		self.fev = 0
		self.elapsed_time = 0
		if max_fevs:
			self.max_fev = max_fevs
		else:
			self.max_fev = None
		self.max_time = max_time
		# Create empty lists
		self.D_list = []
		self.purity_list = []
		self.overlap_list = []
		self.fev_D_list = []
		self.fev_purity_list = []
		self.fev_overlap_list = []
		# Save initial CSS
		p = self.lower_params
		theta, phi = self.upper_params_unpacker(self.upper_params)
		self.css = self.css_density_matrix(p, theta, phi)
		mmwrite(f'{self.directory}/initial_trial_state.mtx', self.css, field='complex', precision=10, symmetry='hermitian')
		# Set up minimizer kwargs
		self.min_kwargs = min_kwargs or dict()
		with open(f'{self.directory}/min_kwargs.json', 'w') as f:
			json.dump(self.min_kwargs, f)
		self.min_kwargs.update(func=self.upper_cost_function, x0=self.upper_params, bounds=self.upper_param_bounds,
		                       callback=self.callback)
		# Start minimization
		self.start_time = time.perf_counter()
		result = self.min_D()
		# Save last CSS
		p = self.lower_params
		theta, phi = self.upper_params_unpacker(self.upper_params)
		self.css = self.css_density_matrix(p, theta, phi)
		mmwrite(f'{self.directory}/calculated_css.mtx', self.css, field='complex', precision=10, symmetry='hermitian')
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


	def callback(self, *args, **kwargs):
		self.iter += 1
		# Update data
		self.D_list.append(self.cost)
		self.purity_list.append(self.purity)
		self.overlap_list.append(self.overlap)

		self.elapsed_time = time.perf_counter() - self.start_time
		print(f"| {self.iter} | {self.fev} | {self.D_list[-1]:.10f} | {self.elapsed_time:.2f} |")

	def min_D(self):
		try:
			# Print headers
			print("| Iterations | Function Evaluations | HSD | Elapsed Time |")
			# Minimize
			# result = dual_annealing(**self.min_kwargs)
			result = NFT(maxfev=2048, reset_interval=16, callback=self.callback) \
				.minimize(fun=self.upper_cost_function, x0=self.upper_params, bounds=self.upper_param_bounds)
		# Break point reached
		except BreakOut:
			return None

		return result

	def upper_cost_function(self, upper_params):
		# Get theta and phi
		theta, phi = self.upper_params_unpacker(upper_params)
		# Calculate purity and overlap lists
		self.pure_states = self.calculate_pure_states(theta, phi)
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
		sigma = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
		for i, j in zip(p, self.pure_states):
			sigma += i * j
		self.purity = np.trace(sigma @ sigma).real
		self.overlap = np.trace(self.test_state @ sigma).real
		return self.test_state_purity + self.purity - 2 * self.overlap

	def cost_function(self, params):
		# Get p, theta and phi
		p, theta, phi = self.params_unpacker(params)
		# Calculate purity and overlap lists
		sigma = self.css_density_matrix(p, theta, phi)
		self.purity = np.trace(sigma @ sigma).real
		self.overlap = np.trace(self.test_state @ sigma).real
		self.cost = self.test_state_purity + self.purity - 2 * self.overlap
		# Update run
		self.updater(p, theta, phi)
		# Return cost
		return self.cost

	def calculate_pure_states(self, theta, phi):
		sigmas = []
		for i in range(self.num_states):
			j = i * self.nqubits
			k = (i + 1) * self.nqubits
			sigmas.append(self.prod_state(theta[j:k], phi[j:k]))

		return sigmas

	def css_density_matrix(self, p, theta, phi):
		sigma = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
		for i in range(self.num_states):
			j = i * self.nqubits
			k = (i + 1) * self.nqubits
			sigma += p[i] * self.prod_state(theta[j:k], phi[j:k])

		return sigma

	def prod_state(self, theta, phi):
		qc = QuantumCircuit(self.nqubits)
		for i, [t, p] in enumerate(zip(theta, phi)):
			qc.add_RY_gate(i, t)
			qc.add_RZ_gate(i, p)
		state = QuantumState(self.nqubits)
		qc.update_quantum_state(state)
		statevector = state.get_vector()
		return np.outer(statevector, statevector.conj())
