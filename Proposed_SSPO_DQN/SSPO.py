import random
import numpy as np

def algm(total_weights=100):  # Accept number of weights from outside
    def fitness_fn(pop):
        return [np.sum(np.array(ind) ** 4) for ind in pop]

    def initialize_population(pop_size, num_weights):
        return [np.random.uniform(-1, 1, num_weights).tolist() for _ in range(pop_size)]

    student = 10  # Population size
    var = total_weights  # Number of weights to optimize
    Max_iteration = 10  # Maximum iterations

    # Initialize the population
    x = initialize_population(student, var)
    
    # Calculate fitness for initial population
    fit = fitness_fn(x)
    Best_fitness = min(fit)
    Best_student = x[np.argmin(fit)]
    
    # Parameters for the algorithm
    a0, b0, bmax = 1, 1, 5
    t = 0

    while t < Max_iteration:
        itr = t + 1
        alpha = a0 - ((a0 / Max_iteration) * itr)  # Update alpha
        beta = b0 + ((bmax - b0) / Max_iteration) * itr  # Update beta
        
        # Calculate the mean of the population
        mean = np.mean(x, axis=0)
        
        # Generate random numbers for decision making
        check = np.random.rand(student)
        mid = np.random.rand(student)

        # Iterate over each student in the population
        for i in range(student):
            # Calculate min and max once for each student
            x_min = np.min(x[i])
            x_max = np.max(x[i])

            for j in range(var):
                if Best_fitness == min(fit):  # If best fitness remains unchanged
                    k = random.randint(1, 2)
                    x[i][j] = Best_student[j] + (-1) ** k * random.random() * (Best_student[j] - x[i][j])
                elif check[i] < mid[i]:
                    rta = random.random()
                    if rta > random.random():
                        x[i][j] = Best_student[j] + random.random() * (Best_student[j] - x[i][j])
                    else:
                        rand = random.random()
                        r = random.random()
                        denom = beta * rand - r
                        if denom != 0:  # Prevent division by zero
                            x[i][j] = (beta * rand / denom) * (
                                ((x[i][j] + r * (alpha * rand * x[i][j] - x[i][j] * (1 - rand * (beta + alpha)))) / (beta * rand)) - mean[j]
                            )
                an = random.random()
                if random.random() > an:
                    x[i][j] += random.random() * (mean[j] - x[i][j])
                else:
                    x[i][j] = x_min + random.random() * (x_max - x_min)  # Use precomputed min and max

        # Recalculate fitness for the new population
        fit = fitness_fn(x)
        Best_student = x[np.argmin(fit)]
        Best_fitness = min(fit)

        t += 1  # Increment iteration count

    return np.array(Best_student).flatten()  # Return the optimized weights as a 1D array
