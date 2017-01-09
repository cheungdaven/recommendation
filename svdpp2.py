import numpy as np
import pandas as pd
from sklearn import cross_validation  as cv
import matplotlib.pyplot as plt
from collections import defaultdict

numFactors = 20


# Load Data
def loadData(path):
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(path, sep='\t', names=header)
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]
    print("Number of users=" + str(n_users) + "; Number of items=" + str(n_items))
    return df,n_users,n_items

# Create training and test matrix
def train_test_split(df,n_users,n_items):
    train_data, test_data = cv.train_test_split(df, test_size=0.25)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)
    R = np.zeros((n_users, n_items))
    length = len(train_data)
    sum = 0
    item_by_users = {}
    for line in train_data.itertuples():
        R[line[1] - 1, line[2] - 1] = line[3]
        sum += line[3]
        item_by_users.setdefault(line[1]-1, []).append(line[2]-1)
    average = float(sum/length)
    print("length="+str(length))
    print("sum="+str(sum))
    T = np.zeros((n_users, n_items))
    for line in test_data.itertuples():
        T[line[1] - 1, line[2] - 1] = line[3]
    return R, T, average,item_by_users

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
    R, T, average, item_by_users= train_test_split(df, n_users, n_items)
    I, I2 = index_matrix(R,T)

    gama1 = 0.001
    gama2 = 0.001
    gama3 = 0.0001
    lambda6 = 0.1
    lambda7 = 0.1
    lambda8 = 0.00001
    # gama1 = 0.0001
    # gama2 = 0.0001
    # gama3 = 0.0001
    # lambda6 = 0.00001
    # lambda7 = 0.00001
    # lambda8 = 0.00001
    k = 10 # number of similar items
    num_epochs = 10

    m, n = R.shape
    users, items = R.nonzero()
    n_u = 0 # number of items rated by u
    train_errors = []
    test_errors = []

    P = np.random.rand(numFactors,m) #latent user feature
    Q = np.random.rand(numFactors,n) #latent movie feature
    Y = np.random.rand(n,numFactors) #accordding to SVDPlusPlusFactorizer.java
    B_U =  np.random.rand(m)
    B_I =  np.random.rand(n)
    #print(type(Q))


    for epoch in range(num_epochs):
        print("epoch=" + str(epoch) + "...........................................................................")
        count = 0
        for u, i in zip(users, items):  # 75000 times
            n_u = len(users[users == u])
            if(R[u,i]>5 or R[u,i]<0):
                print("R["+str(u)+","+str(i)+"]="+R[u,i])
            #print(n_u)
            pPlusY = np.zeros(20)

            for j in item_by_users[u]:
                pPlusY = np.add(pPlusY,Y[j,:])
            #print(pPlusY)
            pPlusY = np.add(pPlusY/np.sqrt(n_u),P[:,u])
            #print(pPlusY)

            #----------------------------------
            # yy = defaultdict(float)
            # for j in item_by_users[u+1]:
            #     for f in range(numFactors):
            #     #     #print(str(i)+"---"+str(f)+":Y[i][f]="+str(Y[i][f]))
            #         yy[f] += Y[i][f]
            # for f in range(len(pPlusY)):
            #     pPlusY[f] = float(pPlusY[f]/np.sqrt(n_u)+P[f][u])
            # print("ddddddddddddddddddddddd")
            # print(yy)
            # ----------------------------------
            error = R[u, i] - (average + B_U[u]+B_I[i]+prediction(pPlusY, Q[:, i]))
            if abs(error) >5:
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++error="+str(error))
            #print(type(error)) float64
            P[:, u] += gama2 * (error * Q[:, i] - lambda7 * P[:, u])
            Q[:, i] += gama2 * (error * (P[:, u]+ 1 / np.sqrt(n_u) *  pPlusY) - lambda7 * Q[:, i])

            # for f in range(numFactors):
            #     for i in item_by_users[u + 1]:
            #         Y[i, f] += gama2 * (error * 1 / np.sqrt(n_u) * Q[f, i] - lambda7 * Y[i][f])

            for item in item_by_users[u]:
                Y[item, :] += gama2 * (error * 1 / np.sqrt(n_u) * Q[:, item] - lambda7 * Y[item,:])
            B_U[u] += gama1 * (error - lambda6 * B_U[u])
            B_I[i] += gama1 * (error - lambda6 * B_I[i])
            print("error="+str(error))
            count += 1
            print("count=========================================="+str(count)+"==================epoch="+str(epoch))
            # print("P[:, u]="+str(P[:, u]))
            # print("Q[:, i]="+str(Q[:, i]))
            # print("B_U[u]="+str(B_U[u]))
            # print("B_I[i]="+str(B_I[i]))
            # print("Y[i, :]="+str(Y[i, :]))
        # train_rmse = rmse(I, R, item_by_users,average, Q, P, Y, B_U, B_I)
        # print("train_rmse="+str(train_rmse))
        # test_rmse = rmse(I2, T, item_by_users,average, Q, P, Y, B_U, B_I)
        # train_errors.append(train_rmse)
        # test_errors.append(test_rmse)
    train_rmse = rmse(I, R, item_by_users, average, Q, P, Y, B_U, B_I)
    test_rmse = rmse(I2, T, item_by_users, average, Q, P, Y, B_U, B_I)
    print("train_rmse=" + str(train_rmse))
    print("test_rmse=" + str(test_rmse))
    #plotRMSE(num_epochs, train_errors, test_errors)



if __name__ == '__main__':
    path = "B:/Datasets/MovieLens/ml-100k/u.data"
    svdpp(path)
    #users, items = R.nonzero()
    #print(item_by_users)
    d = np.random.rand(3,5)
    print(np.sum(d,axis=0))
    print(d[:,1])
