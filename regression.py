import numpy as np
import matplotlib.pyplot as plt

x = np.array([3,4,5,6,8,10,12])
y= np.array([16,12,9.5,8,6,4.5,4])

meansX = np.mean(x)
meansY = np.mean(y)
varX = np.var(x)
varY = np.var(y)

corcoef = np.corrcoef(x,y)
print(corcoef)

#plt.plot(x,y)
#plt.show()



# x et y sont très corrélées, score -0.92

#L_Beta représente l'erreur quadratique moyenne, fonction cout
x = x.reshape(7, 1)
y = y.reshape(7, 1)

X = np.hstack((np.ones(x.shape),x))
print(X)
print(X.shape)

theta = np.random.randn(2,1)

def model(X,theta):
    return X.dot(theta)

print(model(X,theta))

plt.scatter(x,y)
plt.plot(x,model(X,theta,),c='r')
plt.show()

def cost_function(X,y,theta): 
    m = len(y )
    return 1/(2*m) * np.sum((model(X,theta)- y)**2)

print(cost_function(X,y,theta))

# on entraine le modele avec l'algorithme de descente du gradient 

def grad(X,y,theta): 
    m = len(y)
    return 1/m * X.T.dot(model(X,theta) - y)

def gradient_descent(X,y,theta, learning_rate, niteration):
    for i in range(0,niteration):
        theta = theta - learning_rate*grad(X,y,theta)
    return theta

theta_final = gradient_descent(X,y,theta, 0.01, niteration=10000)

print((theta_final))

predictions = model(X,theta_final)
plt.scatter(x,y)
plt.plot(x,predictions)
plt.show()

def coef_determination(y, predictions):
    u=np.sum((y-predictions)**2)
    v=np.sum((y-y.mean())**2)
    return 1 - u/v

print(coef_determination(y, predictions))

#temps d'attente moyen : 

def f(x): 
    return theta_final[1]*x + theta_final[0]

print (f(1)) #1 CAISSE resultat : 16mn
print(f(7)) #2 CAISSES resultat : 8 mn
print(f(20)) #3 CAISSES resultat -7 mn ==> problème du modèle linéaire non adapaté au problème d'un temps d'attente
                    