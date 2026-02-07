import random
import math
import numpy as np

def algm():
    lb, ub = 1, 20
    N, M = 20, 10
    Max_iter = 2

    def bound(value):
        value = int(value)
        if value < lb or value > ub:
            value = random.randint(lb, ub)
        return value

    # Initial solution
    def generate_soln(n, m, Xmin, Xmax):
        data = []
        for _ in range(n):
            row = [random.uniform(Xmin, Xmax) for _ in range(m)]
            data.append(row)
        return np.array(data)

    def fitness(soln):
        hr = random.random()
        fit = [sum([x * hr for x in row]) for row in soln]
        return fit

    Position = generate_soln(N, M, lb, ub)  # initialize population
    Fit = fitness(Position)
    Xbest = Position[np.argmin(Fit)]
    t = 0
    overall_best = []

    while t < Max_iter:
        for i in range(N):
            for j in range(M):
                a = 2 - t * (2 / Max_iter)
                a2 = -1 + t * (-1 / Max_iter)
                r1 = random.random()
                r2 = random.random()
                A = 2 * a * r1 - a
                C = 2 * r2
                b = 1
                l = (a2 - 1) * random.random() + 1
                p = random.random()

                if p < 0.5:
                    if abs(A) < 1:
                        D = abs(C * Xbest[j] - Position[i][j])
                        Position[i][j] = Xbest[j] - A * D
                    else:
                        rand_idx = random.randint(0, N - 1)
                        X_rand = Position[rand_idx]
                        D_rand = abs(C * X_rand[j] - Position[i][j])
                        Position[i][j] = X_rand[j] - A * D_rand
                else:
                    D = abs(Xbest[j] - Position[i][j])
                    Position[i][j] = D * math.exp(b * l) * math.cos(l * 2 * math.pi) + Xbest[j]

        Fit = fitness(Position)
        Xbest = Position[np.argmin(Fit)]
        overall_best = Xbest
        t += 1

    # Return the best solution vector found
    return np.array(overall_best)
