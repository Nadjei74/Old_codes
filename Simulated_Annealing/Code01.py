import time
import random
import math
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Customization section:
initial_temperature = 100
cooling = 0.8  # cooling coefficient
number_variables =4
upper_bounds = [ 21530324, 7728, 13010, 67740]
lower_bounds = [ 16890, 6, 13000, 13000]
computing_time = 1  # second(s)


def objective_function(X):
    x = X[0]
    y = X[1]
    z = X[2]
    g = X[3]
    value = (x/y)*math.log(z/g)
    return value


# ------------------------------------------------------------------------------
# Simulated Annealing Algorithm:
initial_solution = np.zeros((number_variables))
for v in range(number_variables):
    initial_solution[v] = random.uniform(lower_bounds[v], upper_bounds[v])

current_solution = initial_solution
best_solution = initial_solution
n = 1  # no of solutions accepted
best_fitness = objective_function(best_solution)
current_temperature = initial_temperature  # current temperature
start = time.time()
no_attempts = 150  # number of attempts in each level of temperature
record_best_fitness = []
record_best_solution = []
record_i = []
for i in range(9999999):
    for j in range(no_attempts):

        for k in range(number_variables):
            current_solution[k] = best_solution[k] + 0.1 * (random.uniform(lower_bounds[k], upper_bounds[k]))
            current_solution[k] = max(min(current_solution[k],upper_bounds[k]), lower_bounds[k])

        current_fitness=objective_function(current_solution)
        E=abs(current_fitness-best_fitness)
        if i==0 and j==0:
            EA=E

        if current_fitness > best_fitness:
            p=math.exp(-E/(EA*current_temperature))
            print(p)
            if random.random()<p:
                accept= True

            else :
                accept= False

        else:
            accept= True

        if accept == True:
            best_solution=current_solution
            best_fitness=objective_function(best_solution)
            n=n+1
            EA=(EA*(n-1)+E)/n

    print('iteration: {}, best_solution: {}, best_fitness: {}'.format(i, best_solution, best_fitness))
    record_best_fitness.append(best_fitness)
    current_temperature=current_temperature*cooling
    end= time.time()
    if end-start>= computing_time:
        break


plt.plot(record_best_fitness)
plt.show()
