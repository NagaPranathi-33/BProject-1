import random, math, numpy as np

def algm():
    Final_best = []
    Max_Generation = 20
    N = 4          # row size
    D = 5           # column size
    G = 2           # time step between(2, 20)
    Solution_update =[]

    def func(solution):
        fit = []
        for i in range(len(solution)):
            a = 0.0
            for j in range(len(solution[i])):
                a += solution[i][j]
            fit.append(a)
        return fit

    def Solution(N,D):
        Solution = []
        for i in range(N):
            tem = []
            for j in range(D):
                a = random.random()
                tem.append(a)
            Solution.append(tem)
        return Solution


    Solution=Solution(N,D)
    #solution update
    def update_solution(N, D, Fit, rooster, hen, chick, Solution):
        update_soln = []
        for i in range(N):
            tem = []
            for j in range(D):
                epsilon = 0.1  # ep.. => small constant to avoid zero
            if (i in rooster):
                tem = update_rooster(N, epsilon, Fit, Solution)
            if (i in hen):
                tem = update_hen(N, epsilon, Fit, Solution)
            if (i in chick):
                tem = update_chick(N, Solution)
            update_soln.append(tem)
        return update_soln

    def better_solution(Solution, Solution_update):
        best_soln = []
        soln_fit = func(Solution)  # Fitness of solution
        soln_up_fit = func(Solution_update)
        # for each row solution with min. fitness_CSO is taken as best_solution
        for i in range(len(Solution)):
            if (soln_fit[i] < soln_up_fit[i]):
                best_soln.append(Solution[i])
            else:
                best_soln.append(Solution_update[i])
        return best_soln  # (each Solution with min fitness_CSO)

    def update_rooster(N, epsilon, Fit, Solution):
        tem = []
        #k value belongs to i, but != i
        if (i < N - 1):
            k = i + 1 #(here k = i+1)
        else:
            k = 0
        # sigma _sq => standard deviation
        if (Fit[i] <= Fit[k]):
            sigma_sq = 1.0
        else:
            sigma_sq = math.exp((Fit[k] - Fit[i]) / abs((Fit[i] + epsilon)))

        for j in range(len(Solution[i])):
            r = Solution[i][j] * (1 + random.random() * sigma_sq)
            tem.append(r)
        return (tem)


    def update_hen(N, epsilon, Fit, Solution):
        # Rand => random no. b/w 0 & 1, r1 & r2 belongs to i but r1 != r2
        Rand = random.random()
        tem = []
        if (i < N - 1):
            r1 = i + 1
            if (i == N - 2):
                r2 = 0
            else:
                r2 = r1 + 1
        else:
            r1 = 0
            r2 = r1 + 1

        # S1 & S2
        s1 = math.exp((Fit[i] - Fit[r1])) / abs(Fit[i] + epsilon)
        s2 = math.exp(Fit[r2] - Fit[i])

        for j in range(len(Solution[i])):
            h = Solution[i][j] + (
                    (s1 * Rand * Solution[r1][j] - Solution[i][j]) + (s2 * Rand * Solution[r2][j] - Solution[i][j]))
            tem.append(abs(h))
        return tem


    def update_chick(N, Solution):
        tem = []
        # FL => random between (0,2), m belongs to i
        FL = random.randint(1, 2)
        if (i < (N - 1)):
            m = i + 1
        else:
            m = 0
            # update
        for j in range(len(Solution[i])):
            c = abs(Solution[i][j] + (FL * Solution[m][j] - Solution[i][j]))
            tem.append(abs(c))
        return tem
            # end update chick

    t = 0
    rooster = []
    hen = []
    chick = []
    Fit = []
    Fit_tem1 = []
    Fit_tem2 = []
    nXL = np.argmax(func(Solution))  # finding the position of maximum value in the fitness
    XL = Solution[nXL]  # asking to print the row of X which is having the maximum value of fitness(finding the leading rider)
    best_fit = np.max(func(Solution))
    # print(XL)

    #loop
    while(t <  Max_Generation):
        #status only updated in G time step (0%n = 0--so update will done in 1st iteration)
        if(t % G ==0):
            rooster = []
            hen = []
            chick = []

            #Fitness calculation
            Fit_tem1 = []
            Fit_tem2 = []
            Fit = func(Solution)
            Fit_tem1 = Fit.copy()       # for best -- clone fitness_CSO (to group swarm)
            Fit_tem2 = Fit.copy()       # for worst
            #  DIVIDE THE SWARM into groups(Rooster(best), Hen, Chick(worst)) -- based on fitness_CSO(best = min)
            m = int(N/3)
            for i in range(m):
                #rooster --best m
                r = Fit_tem1.index(min(Fit_tem1))
                rooster.append(r)
                Fit_tem1[r] = 10000
                #chick--worst m
                c = Fit_tem2.index(max(Fit_tem2))
                chick.append(c)
                Fit_tem2[c] = 0

            # hen others
            for i in range(N):
                if(i not in rooster) and  (i not in chick):
                    hen.append(i)


        Solution_update = update_solution(N, D, Fit, rooster, hen, chick, Solution)
        # end better soln
        # end class
        Solution = better_solution(Solution, Solution_update)
        new_fit = func(Solution)

        NF = np.argmin(new_fit)  # the position of maximum of newfit
        nXL = np.argmin(new_fit)  # finding the position of maximum value in the fitness
        if (np.min(new_fit) < np.min(Fit)):
            XL = Solution[NF]
            best_fit = np.min(new_fit)
        Fit = new_fit.copy()

        t += 1
    #end while
    Fit = func(Solution)
    Final_best = Solution[Fit.index(min(Fit))]
    return np.max(Final_best)










