import numpy as np
import pandas as pd
from sklearn import cross_validation  as cv
import matplotlib.pyplot as plt

'''
Author: Shuai Zhang
Website: http://shuaizhang.tech
E-mail: cheungshuai@outlook.com
Feel free to contact me if you have any problem.
'''

numFactors =  20


# Load Data
def loadData(path):
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(path, sep='\t', names=header)
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]
    return df,n_users,n_items

# Create training and test matrix
def train_test_split(df,n_users,n_items):
    train_data, test_data = cv.train_test_split(df, test_size=0.2)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)
    R = np.zeros((n_users, n_items))
    length_train = len(train_data)
    length_test = len(test_data)
    sum_train = 0
    sum_test = 0
    item_by_users = {}
    users_by_item = {}
    rate_by_users = {}
    for line in train_data.itertuples():
        R[line[1] - 1, line[2] - 1] = line[3]
        sum_train += line[3]
        item_by_users.setdefault(line[1]-1, []).append(line[2]-1)
        users_by_item.setdefault(line[2]-1, []).append(line[1]-1)
        rate_by_users.setdefault(line[1]-1, []).append(line[3])
    average_train = float(sum_train/length_train)
    T = np.zeros((n_users, n_items))
    for line in test_data.itertuples():
        T[line[1] - 1, line[2] - 1] = line[3]
        sum_test += line[3]
    average_test = float(sum_test/length_test)
    return R, T, average_train,average_test,item_by_users,users_by_item,rate_by_users

# Index matrix for training data
def index_matrix(R,T):
    I = R.copy()
    I[I > 0] = 1
    I[I == 0] = 0
    I2 = T.copy()
    I2[I2 > 0] = 1
    I2[I2 == 0] = 0
    return I, I2

def prediction(P,Q):
    #print(np.shape(1/np.sqrt(n_u)*np.sum(Y)))
    #print(1 / np.sqrt(n_u) * np.sum(Y))
    return np.dot(P.T,Q)

def rmse(I, R, item_by_users,average, Q, P, Y, B_U, B_I):
    users, items = R.nonzero()
    sum = 0
    for u, i in zip(users, items):  # 75000 times
        n_u = len(users[users == u])
        if (R[u, i] > 5 or R[u, i] < 0):
            print("R[" + str(u) + "," + str(i) + "]=" + R[u, i])
        # print(n_u)
        pPlusY = np.zeros(numFactors)
        for j in item_by_users[u]:
            pPlusY = np.add(pPlusY, Y[j, :])
        # print(pPlusY)
        pPlusY = np.add(pPlusY / np.sqrt(n_u), P[:, u])
        error = R[u, i] - (average + B_U[u] + B_I[i] + prediction(pPlusY, Q[:, i]))
        sum += error**2
    return np.sqrt(sum/len(R[R > 0]))

def plotRMSE(n_epochs,train_errors,test_errors):
    plt.plot(range(n_epochs), train_errors, marker='o', label='Training Data')
    plt.plot(range(n_epochs), test_errors, marker='v', label='Test_data')
    plt.title('SGD-WR Learning Curve')
    plt.xlabel('Number of Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    plt.show()

def svdpp(path):
    df, n_users, n_items = loadData(path)
    R, T, average_train, average_test, item_by_users, users_by_item, rate_by_users= train_test_split(df, n_users, n_items)
    I, I2 = index_matrix(R,T)

    gama1 = 0.01
    gama2 = 0.01
    lambda6 = 0.05
    lambda7 = 0.1
    num_epochs = 100

    m, n = R.shape
    users, items = R.nonzero()
    train_errors = []
    test_errors = []

    P = np.random.rand(numFactors,m) #latent user feature
    Q = np.random.rand(numFactors,n) #latent movie feature
    Y = np.random.rand(n,numFactors) #accordding to SVDPlusPlusFactorizer.java
    B_U =  np.random.rand(m)
    B_I =  np.random.rand(n)

    pPlusY = {}

    for u in range(m):
        p = np.zeros(numFactors)
        for j in item_by_users[u]:
            p = np.add(p,Y[j,:])
        pPlusY[u] = p

    for epoch in range(num_epochs):
        print("epoch=" + str(epoch))
        count = 0
        for u, i in zip(users, items):  # 75000 times
            n_u = len(users[users == u])
            pPlusY[u] = np.add(pPlusY[u] / np.sqrt(n_u), P[:, u])
            error = R[u, i] - (average_train + B_U[u]+B_I[i]+prediction( pPlusY[u], Q[:, i]))
            print("Error="+str(error))
            P[:, u] += gama2 * (error * Q[:, i] - lambda7 * P[:, u])
            Q[:, i] += gama2 * (error * (P[:, u]+ 1 / np.sqrt(n_u) *   pPlusY[u]) - lambda7 * Q[:, i])

            for item in item_by_users[u]:
                Y[item, :] += gama2 * (error * 1 / np.sqrt(n_u) * Q[:, item] - lambda7 * Y[item,:])
            B_U[u] += gama1 * (error - lambda6 * B_U[u])
            B_I[i] += gama1 * (error - lambda6 * B_I[i])
            count += 1

        train_rmse = rmse(I, R, item_by_users,average_train, Q, P, Y, B_U, B_I)
        print("train_rmse="+str(train_rmse))
        test_rmse = rmse(I2, T, item_by_users,average_train, Q, P, Y, B_U, B_I)
        print("test_rmse=" + str(test_rmse))
        train_errors.append(train_rmse)
        test_errors.append(test_rmse)

    print("train_errors=" + str(train_errors))
    print("test_errors=" + str(test_errors))
    plotRMSE(num_epochs, train_errors, test_errors)



if __name__ == '__main__':
    path = "B:/Datasets/MovieLens/ml-100k/u.data"
    svdpp(path)
