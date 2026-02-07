import random, math, numpy as np


def clustering(data,n_c,lab):
    U, obj_fcn = [], []

    def initfcm(cluster_n, data_n):
        U_part, U_transpose = [], []
        col_sum, col_sum_change = [], []
        for i in range(cluster_n):
            tem = []
            for j in range(data_n):
                tem.append(random.random())
            U_part.append(tem)

        U_transpose = np.transpose(U_part)
        for i in range(len(U_transpose)):
            sum = 0
            for j in range(len(U_transpose[i])):
                sum += U_transpose[i][j]
            col_sum.append(sum)

        for i in range(cluster_n):
            col_sum_change.append(col_sum)  # make col_sum size = U_part size = cluster size

        # elementwise division
        for i in range(len(U_part)):
            tem = []
            for j in range(len(U_part[i])):
                tem.append(U_part[i][j] / col_sum_change[i][j])
            U.append(tem)
        return U


    def distfcm(center, data):
        out, ones, matmul_ones_center, data_minus_ones = [], [] ,[], []

        # fill the output matrix
        for i in range(len(data)):
            tem = []
            for j in range(1):
                tem.append(1.0)
            ones.append(tem)        # ones of size data size x 1

        for k in range(len(center)):
            # matrix mul of ones and center or copy each row of center to data size(to make center size = data size)
            matmul_ones_center = []
            for i in range(len(data)):
                matmul_ones_center.append(center[k])

            # (data - matmul_ones_center) ^ 2 -- (elementwise)
            data_minus_ones = []
            for i in range(len(data)):
                tem = []
                for j in range(len(data[i])):
                    tem.append(math.pow((data[i][j] - matmul_ones_center[i][j]), 2))
                data_minus_ones.append(tem)

            # sum elements of row, then take square root
            tem = []
            for i in range(len(data_minus_ones)):
                sum = 0
                for j in range(len(data_minus_ones[i])):
                    sum += data_minus_ones[i][j]
                tem.append(math.sqrt(sum))
            out.append(tem)

        return out

    def stepfcm(data, U, cluster_n, expo):
        tmp_sum, mf, matmul_mf_data, matmul_row_mf_sum_ones, row_mf_sum = [], [], [], [], []
        ones, center, dist, dist_mul_mf, tmp_sum_change, U_new = [], [], [], [], [], []

        for i in range(len(U)):
            tem = []
            for j in range(len(U[i])):
                tem.append(math.pow(U[i][j], expo))
            mf.append(tem)

        # mf * data
        for i in range(len(mf)):
            tem = []
            for j in range(len(data[0])):
                a = 0
                for k in range(len(data)):
                    a += (mf[i][k] * data[k][j])    # matrix mul of mf and data
                tem.append(a)
            matmul_mf_data.append(tem)

        for i in range(len(mf)):
            tem, sum = [], 0
            for j in range(len(mf[i])):
                sum += mf[i][j]             # adding attributes of the row to make single column
            tem.append(sum)
            row_mf_sum.append(tem)

        for i in range(1):
            tem = []
            for j in range(len(data[0])):
                tem.append(1.0)
            ones.append(tem)

        # row_mf_sum * ones
        for i in range(len(row_mf_sum)):
            tem = []
            for j in range(len(ones[0])):
                a = 0
                for k in range(len(ones)):
                    a += (row_mf_sum[i][k] * ones[k][j])    # matrix mul of row_mf_sum and ones
                tem.append(a)
            matmul_row_mf_sum_ones.append(tem)

        # elementwise division
        for i in range(len(matmul_mf_data)):
            tem = []
            for j in range(len(matmul_mf_data[i])):
                tem.append(matmul_mf_data[i][j]/matmul_row_mf_sum_ones[i][j])
            center.append(tem)

        dist = distfcm(center, data)

        a = 0
        for i in range(len(dist)):
            tem = []
            for j in range(len(dist[i])):
                a += math.pow(dist[i][j], 2) * mf[i][j]
        obj_fcn.append(a)

        tmp, tmp_trans = [], []
        for i in range(len(dist)):
            tem = []
            for j in range(len(dist[i])):
                tem.append(math.pow(dist[i][j], (-2/(expo-1)))) # tmp = dist ^ (-2/(expo-1))
            tmp.append(tem)
        tmp_trans = np.transpose(tmp)

        for i in range(len(tmp_trans)):
            sum = 0
            for j in range(len(tmp_trans[i])):
                sum += tmp_trans[i][j]
            tmp_sum.append(sum)

        for i in range(cluster_n):
            tmp_sum_change.append(tmp_sum)      # make tmp_sum size= cluster size

        for i in range(len(tmp)):
            tem = []
            for j in range(len(tmp[i])):
                tem.append(tmp[i][j] / tmp_sum_change[i][j])
            U_new.append(tem)

        return U_new

    U_trans, Cluster = [], []
    data_n, cluster_n = len(data), n_c            # data, cluster size
    expo, max_iter, min_impro = 2, 100, 1e-5    # exponent of U, max iteration, min improvement
    U = initfcm(cluster_n, data_n)              # Initial fuzzy partition
    g = 0

    while (g < max_iter):
        U = stepfcm(data, U, cluster_n, expo)
        if(g > 1):
            if(np.abs(obj_fcn[g] - obj_fcn[g-1]) < min_impro):
                g = max_iter
        g += 1

    U_trans = np.transpose(U)
    U_trans = U_trans.tolist()
    for i in range(len(U_trans)):
        random.shuffle(U_trans[i])
    for i in range(len(U_trans)):
        Cluster.append(np.argmax(U_trans[i]))  # clusters

    ind = [i for i in range(n_c)]  # Create index
    cluster_g = [[] for i in range(n_c)]  # Create empty bracket list
    lab_g = [[] for i in range(n_c)]  # Create empty bracket list
    for i in range(len(Cluster)):
        for j in range(len(ind)):
            if Cluster[i] == ind[j]:
                cluster_g[ind[j]].append(data[i])
                lab_g[ind[j]].append(lab[i])
    return cluster_g, lab_g