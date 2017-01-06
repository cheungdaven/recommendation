from math import *
from numpy import *

'''
written by Shuai Zhang, http://shuaizhang.tech/
implementation of neighborhood and SVD++ integrated model
refer to paper: Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model. Yehuda Koren, 2008
'''
path = 'B:/Datasets/MovieLens/ml-100k/u.data'
user_path = 'B:/Datasets/MovieLens/ml-100k/u.user'
item_path = 'B:/Datasets/MovieLens/ml-100k/u.item'

def loadMovieLens100K(path):
    fr = open(path)
    ratingMatrix = []
    for line in fr.readlines():
        lineArr = line.strip().split()
        ratingMatrix.append([int(lineArr[0]), int(lineArr[1]), int(lineArr[2]), int(lineArr[3])])
        #print([int(lineArr[0]), int(lineArr[1]), int(lineArr[2]), int(lineArr[3])])
    fr.close()
    return ratingMatrix

def convert2Matrix(ratingMatrix, userMatrix, itemMatrix):
    '''
    convert rating to real matrix
    :param ratingMatrix:
    :return:
    '''
    matrix = []
    dense_matrix = []
    for u in userMatrix:
        row = sorted(getRu(ratingMatrix, u[0]))
        row_item = [x[1] for x in row]
        row_rate = [x[2] for x in row]
        dense_matrix.append(row_rate)
        row_fill_zero = []
        for i in itemMatrix:
            row_fill_zero += [row_rate[row_item.index(i[0])] if i[0] in row_item else 0]
        print(row_fill_zero)

        matrix.append(row_fill_zero)
    return mat(matrix), dense_matrix


def loadUser(path):
    fr = open(path)
    userMatrix = []
    for line in fr.readlines():
        lineArr = line.strip().split("|")
        userMatrix.append([int(lineArr[0]), int(lineArr[1]), str(lineArr[2]), str(lineArr[3]), str(lineArr[4])])
        #print([int(lineArr[0]), int(lineArr[1]), str(lineArr[2]), str(lineArr[3]), str(lineArr[4])])
    fr.close()
    return userMatrix

def loadItem(path):
    fr = open(path)
    itemMatrix = []
    for line in fr.readlines():
        lineArr = line.strip().split("|")
        itemMatrix.append([int(lineArr[0]), str(lineArr[1]), str(lineArr[2]), str(lineArr[3]), str(lineArr[4])])
        #print([int(lineArr[0]), int(lineArr[1]), str(lineArr[2]), str(lineArr[3]), str(lineArr[4])])
    fr.close()
    return itemMatrix

def getOverAllAverage(ratingMatrix):
    '''
    get the over all average rating
    '''
    size = len(ratingMatrix)
    print(size)
    all = 0
    for r in ratingMatrix:
        all += r[2]
    print(all)
    print(float(all/size))
    return float(all/size)

def getRu(ratingMatrix, u):
    '''
    R(u), contains all the items for which ratings by u are available
    '''
    Ru = []
    for r in ratingMatrix:
        if u == r[0]:
            Ru.append(r)
    return Ru

def getRuSize(dense_matrix, u):
    ratingbyu = dense_matrix[u - 1]
    return len(ratingbyu)

def getAverageRatingByu(dense_matrix, u):
    '''
    get the average rating of u
    :return: u_rating
    '''
    # ratingbyu = matrix[u-1][0].tolist()
    # ratingbyu_remove_zero = [x for x in ratingbyu[0] if x!=0]
    # return float(sum(ratingbyu_remove_zero)/len(ratingbyu_remove_zero))
    ratingbyu = dense_matrix[u-1]
    return float(sum(ratingbyu)/len(ratingbyu))


def getRatingUI(matrix, u_id, i):
    '''
    get rating by user u to item i
    :param ratingMatrix:
    :param u:
    :param i:
    :return:
    '''
    return matrix[u_id-1,i-1]

def similarityIJ(matrix, dense_matrix,userMatrix, i, j):
    '''
    compute the similarity between item i and j, return top k similar items
    using adjust cosine similarity algorithm
    :param ratingMatrix:
    :param i:
    :param j:
    :param k: default 30. works well for 100-k movieLens dataset
    :return:
    '''
    similarity = 0
    numerator = 0
    denominatori = 0
    denominatorj = 0
    for u in userMatrix:
        u_average_rating = getAverageRatingByu(dense_matrix, u[0])
        #print(u_average_rating)
        Rui = getRatingUI(matrix, u[0], i)
        Ruj = getRatingUI(matrix, u[0], j)
        #Rui - u_average_rating
        temp1 = Rui - u_average_rating
        #Ruj - u_average_rating
        temp2 = Ruj - u_average_rating
        numerator += temp1 * temp2
        denominatori += temp1**2
        denominatorj += temp2**2

    similarity = float(numerator/sqrt(denominatori * denominatorj))

    return (similarity,j)

def getSki(matrix,dense_matrix,userMatrix, itemMatrix, i, k=30):
    '''
    get set of k items most similar to i
    :param ratingMatrix:
    :param i:
    :param k:
    :return:
    '''
    similarities = []
    for item in itemMatrix :
        if item[0] != i:
            similarities.append(similarityIJ(matrix, dense_matrix, userMatrix, i, item[0]))
    similarities.sort(reverse=True)
    print(similarities[0:k])
    result = [i[1] for i in  similarities[0:k]]
    print(result)
    return result

def getNu(ratingMatrix, u):
    '''
    N(u), contains all items for which user u provided an implicit feedback
    in the simple binary case, it's the same as R(u)
    '''
    Nu = []
    for r in ratingMatrix:
        if u == r[0]:
            Nu.append(r)
    return Nu

def getRkiu(Nu, Ski):
    '''
    compue N^K(i;u) and R^k(i;u), in our case, two of them equals
    :param Nu:
    :param Ski:
    :return:
    '''
    Nu_temp = [i[1] for i in Nu]
    return list(set(Nu_temp).union(set(Ski)))


def svdpp(numIter=200, numFactors=200):
    gama1 = 0.007
    gama2 = 0.007
    gama3 = 0.001
    lambda6 = 0.005
    lambda7 = 0.015
    lambda8 = 0.015
    k = 30
    userMatrix = loadUser(user_path)
    itemMatrix = loadItem(item_path)
    ratingMatrix = loadMovieLens100K(path)
    overall_average = getOverAllAverage(ratingMatrix)



    matrix, dense_matrix = convert2Matrix(ratingMatrix,userMatrix, itemMatrix)

    #print(getAverageRatingByu(dense_matrix,1))
    #Ski = getSki(matrix,dense_matrix,userMatrix,itemMatrix, 2)
    # print("ski:"+Ski)
    # print(getRatingUI(ratingMatrix,256,127))
    # print(Ru_len)
    #Rkiu = getRkiu(Ru, Ski)
    #print(Rkiu)

    M ,N = shape(matrix)
    print("size of users:"+str(M))
    print("size of items:"+str(N))
    #initial values
    b_u = random.uniform(0, 0.1)
    b_i = random.uniform(0, 0.1)
    q_i = random.uniform(0, 0.1,numFactors)
    p_u = random.uniform(0, 0.1,numFactors)
    y_i = random.uniform(0, 0.1,numFactors)

    print(q_i)
    print("q_i = "+str(len(q_i)))


    for iteration in range(numIter):
        for m in range(M):
            if m % 50 == 0:
                print("m="+str(m))
            #Ru = getRu(ratingMatrix, m+1)
            Ru_len = getRuSize(dense_matrix, m+1)
            #print("Ru===" + str(Ru))
            #print("Ru_len===" + str(Ru_len))
            y_i_temp = 1 / sqrt(Ru_len) * y_i * Ru_len
            #print("y_i_temp = " + str(y_i_temp))
            for n in range(N):
                r_ui = getRatingUI(matrix, m,n)
                error_ui = 0
                if r_ui != 0:
                    #print("b_u = " + str(b_u))
                    #print("b_i = " + str(b_i))
                    #print("q_i = " + str(q_i.transpose().dot(p_u + y_i_temp)))
                    #print("p_u = " + str(p_u))
                    #print("y_i = " + str(y_i))
                    r_ui_evaluation = overall_average + b_u + b_i + q_i.transpose().dot(p_u + y_i_temp)
                    print("r_ui_evaluation="+str(r_ui_evaluation))

                    error_ui = r_ui - r_ui_evaluation
                    print("error_ui=" + str(error_ui))
                    #break
                    if abs(error_ui) < 0.01:
                        print("error_ui="+str(error_ui))
                        print("m=" + str(m))
                        #return

                    b_u += gama1 * (error_ui - lambda6 * b_u)
                    b_i += gama1 * (error_ui - lambda6 * b_i)
                    q_i += gama2 * (error_ui * (p_u + 1 / sqrt(Ru_len) * y_i * Ru_len ) -lambda7 * q_i)
                p_u += gama2 * (error_ui * q_i - lambda7 * p_u)
                y_i += gama2 * (error_ui * 1 / sqrt(Ru_len) * q_i - lambda7 * y_i)
        # dataMatrix = mat(dataMatIn)
        # labelMat = mat(classLabels).transpose()
        # m, n = shape(dataMatrix)
        # print(m, n)
        # alpha = 0.001
        # maxCycles = 500
        # weights = ones((n, 1))
        # for k in range(maxCycles):
        #     h = sigmoid(dataMatrix * weights)
        #     error = labelMat - h
        #     weights = weights + alpha * dataMatrix.transpose() * error  # matrix multiplication
        # return weights
    print("result------")
    print("b_u = " + str(b_u))
    print("b_i = " + str(b_i))
    print("q_i = " + str(q_i))
    print("p_u = " + str(p_u))
    print("y_i = " + str(y_i))
    return b_u, b_i, q_i, p_u, y_i

if __name__ == '__main__':
    svdpp()

