import numpy as np
import pandas as pd
from sklearn import cross_validation  as cv
import matplotlib.pyplot as plt

'''
I am not the author of this code
Please refer to the following link:
http://online.cambridgecoding.com/notebooks/mhaller/implementing-your-own-recommender-systems-in-python-using-stochastic-gradient-descent-4
'''


header = ['user_id','item_id', 'rating', 'timestamp']
df = pd.read_csv("B:/Datasets/MovieLens/ml-100k/u.data", sep='\t', names=header)
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print("Number of users="+str(n_users)+"; Number of items="+str(n_items))

train_data, test_data = cv.train_test_split(df, test_size=0.25)
train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)

R = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    R[line[1]-1, line[2]-1] = line[3]
T = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    T[line[1]-1, line[2]-1] = line[3]
#print(R)

I = R.copy()
I[I > 0] = 1
I[I == 0 ] = 0
#print(I)

I2 = T.copy()
I2[I2 > 0] = 1
I2[I2 == 0 ] = 0

def prediction(P,Q):
    return np.dot(P.T,Q)

lmbda = 0.1
k = 20
m,n = R.shape
n_epochs = 50
gamma = 0.01

P = 3 * np.random.rand(k,m)
Q = 3 * np.random.rand(k,n)

#print(P)
#print(Q)

def rmse(I, R, Q, P):
    return np.sqrt(np.sum((I*(R - prediction(P,Q)))**2/len(R[R > 0])))

train_errors = []
test_errors = []

users, items = R.nonzero()
#print(len(users)) #75000
#print(items) #75000


for epoch in range(n_epochs):
    print("epoch="+str(epoch)+"......")
    for u, i in zip(users,items): #75000 times
        #print("i="+str(i))
        #print("u="+str(u))
        e = R[u,i] - prediction(P[:,u], Q[:,i])
        P[:,u] += gamma * (e * Q[:,i] - lmbda * P[:,u])
        Q[:,i] += gamma * (e * P[:,u] - lmbda * Q[:,i])
    train_rmse = rmse(I, R, Q , P)
    test_rmse = rmse(I2, T, Q, P)
    train_errors.append(train_rmse)
    test_errors.append(test_rmse)

print("train_errors=" + str(train_errors))
print("test_errors=" + str(test_errors))

plt.plot(range(n_epochs), train_errors, marker='o', label='Training Data')
plt.plot(range(n_epochs), test_errors, marker='v', label='Test_data')
plt.title('SGD-WR Learning Curve')
plt.xlabel('Number of Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.grid()
plt.show()






