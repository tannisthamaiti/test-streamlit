import numpy as np
import random
from matplotlib import pyplot as plt


def objective(x, y):
    return x**2.0 + y**2.0
 
# derivative of objective function
def derivative(x, y):
    return np.asarray([x * 2.0, y * 2.0])
    
    
# gradient descent algorithm
def gradient_descent(objective, derivative, bounds, n_iter, step_size, momentum):
	# track all solutions
    solutions = list()
    # generate an initial point
    x = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    score = objective(x[0], x[1])
	# keep track of the change
    change = 0.0
	# run the gradient descent
    for t in range(n_iter):
		# calculate gradient
        gradient = derivative(x[0], x[1])
        for i in range(x.shape[0]):
		# calculate update
            new_change = step_size * gradient[i] + momentum * change
            
		# take a step
            x[i] = x[i] - new_change
		# save the change
            change = new_change
		# evaluate candidate point
        
            solution_eval = objective(x[0], x[1])
		# store solution
            solutions.append(x.copy())
            #scores.append(solution_eval)
		# report progress
        print('>%d f(%s) = %.5f' % (t, x, solution_eval))
    return solutions

# gradient descent algorithm with adam
def adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2, eps=1e-8):
 # generate an initial point
    solutions = list()
    x = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    score = objective(x[0], x[1])
 # initialize first and second moments
    m = [0.0 for _ in range(bounds.shape[0])]
    v = [0.0 for _ in range(bounds.shape[0])]
 # run the gradient descent updates
    for t in range(n_iter):
 # calculate gradient g(t)
        g = derivative(x[0], x[1])
     # build a solution one variable at a time
        for i in range(x.shape[0]):
         # m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
            m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
         # v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
            v[i] = beta2 * v[i] + (1.0 - beta2) * g[i]**2
         # mhat(t) = m(t) / (1 - beta1(t))
            mhat = m[i] / (1.0 - beta1**(t+1))
         # vhat(t) = v(t) / (1 - beta2(t))
            vhat = v[i] / (1.0 - beta2**(t+1))
         # x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + eps)
            x[i] = x[i] - alpha * mhat / (np.sqrt(vhat) + eps)
         # evaluate candidate point
            score = objective(x[0], x[1])
            solutions.append(x.copy())
         # report progress
            print('>%d f(%s) = %.5f' % (t, x, score))
    return solutions
    
# gradient descent algorithm with rmsprop
def rmsprop(objective, derivative, bounds, n_iter, step_size, rho):
 # track all solutions
    solutions = list()
     # generate an initial point
    x = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    score = objective(x[0], x[1])
 # list of the average square gradients for each variable
    sq_grad_avg = [0.0 for _ in range(bounds.shape[0])]
 # run the gradient descent
    for it in range(n_iter):
 # calculate gradient
        gradient = derivative(x[0], x[1])
 # update the average of the squared partial derivatives
        for j in range(gradient.shape[0]):
 # calculate the squared gradient
            sg = gradient[j]**2.0
 # update the moving average of the squared gradient
            sq_grad_avg[j] = (sq_grad_avg[j] * rho) + (sg * (1.0-rho))# build solution
            new_solution = list()
            for i in range(x.shape[0]):# calculate the learning rate for this variable
                alpha = step_size / (1e-8 + np.sqrt(sq_grad_avg[j]))
 # calculate the new position in this variable
                value = x[i] - alpha * gradient[j]
                new_solution.append(value)# store the new solution
                solution = np.asarray(new_solution)
        solutions.append(solution)
 # evaluate candidate point
        solution_eval = objective(solution[0], solution[1])# report progress
        print('>%d f(%s) = %.5f' % (it, solution, solution_eval))
    return solutions


# gradient descent algorithm with adamax
def adamax(objective, derivative, bounds, n_iter, alpha, beta1, beta2):
	solutions = list()
	# generate an initial point
	x = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	# initialize moment vector and weighted infinity norm
	m = [0.0 for _ in range(bounds.shape[0])]
	u = [0.0 for _ in range(bounds.shape[0])]
	# run iterations of gradient descent
	for t in range(n_iter):
		# calculate gradient g(t)
		g = derivative(x[0], x[1])
		# build a solution one variable at a time
		for i in range(x.shape[0]):
			# m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
			m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
			# u(t) = max(beta2 * u(t-1), abs(g(t)))
			u[i] = max(beta2 * u[i], abs(g[i]))
			# step_size(t) = alpha / (1 - beta1(t))
			step_size = alpha / (1.0 - beta1**(t+1))
			# delta(t) = m(t) / u(t)
			delta = m[i] / u[i]
			# x(t) = x(t-1) - step_size(t) * delta(t)
			x[i] = x[i] - step_size * delta
		# evaluate candidate point
		score = objective(x[0], x[1])
		solutions.append(x.copy())
		# report progress
		print('>%d f(%s) = %.5f' % (t, x, score))
	return solutions



