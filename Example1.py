import numpy as np
import matplotlib.pyplot as plt

# parameters
sets = 1
points = 20
dimensions = 2
lamb = np.sqrt(np.pi / 8)

# sets = 1
# points = 2
# dimensions = 2

# Generate a 1D array of 25 random data points with a uniform distribution between -3 and 3
data = np.random.uniform(-3, 3, (sets, points, dimensions))
# data = np.array([[[1, 1], [-1, 0]], [[1, -1], [0, 1]]])
#print(data)

# assign labels to each data point
label = np.empty((sets, points))
for n in range(sets):
    for m in range(points):
        x = data[n][m]
        val = np.dot(x, np.array([1, 1]))
        if val > 0:
            label[n][m] = 1
        else:
            label[n][m] = 0

#print(label)

# initialize weights
def initial_w(mu_w, c_w):
    w0 = np.random.normal(mu_w, c_w)
    return w0

# sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# train model using data from set n
def train(n):
    # initialize values
    mu_w = np.array([-1, 0]).T
    sigma_w = np.array([[1, 0], [0, 1]])
    w = initial_w(mu_w, sigma_w)
    mu_z = 0
    var_z = 0
    mu_y = 0
    var_y = 0
    l = 0
    for m in range(points):
        x = data[n][m]
        y = label[n][m]
        z = np.dot(x, w)

        mu_z = np.dot(x, mu_w)
        # print('mu_z' + str(m))
        # print(mu_z)
        var_z = np.dot(np.matmul(x, sigma_w), x.T)
        #print('var_z' + str(m))
        #print(var_z)
        l = np.matmul(sigma_w, x.T) / var_z

        t = np.sqrt(1 + lamb ** 2 * var_z)
        mu_y = sigmoid(mu_z / t)
        #print('mu_y' + str(m))
        #print(mu_y)
        var_y = mu_y * (1 - mu_y) * (1 - (1 / t))
        #print('var_y' + str(m))
        #print(var_y)
        var_zy = (var_z / t) * mu_y * (1 - mu_y)

        # update values
        temp_mu_z = mu_z + (var_zy / var_y) * (y - mu_y)
        temp_var_z = var_z - (var_zy / var_y) * var_zy
        # print('temp_var_z' + str(m))
        # print(temp_var_z)
        mu_w = mu_w + l * (temp_mu_z - mu_z)
        # print('mu_w' + str(m))
        # print(mu_w)
        lc = l * (temp_var_z - var_z)
        inc = np.array([lc]).T @ np.array([l])
        sigma_w = sigma_w + inc
        # print('sigma_w'  + str(m))
        # print(sigma_w)
    return (mu_w, sigma_w, mu_y, var_y)

# train program
for n in range(sets):
    mu_w, sigma_w, mu_y, var_y = train(n)
    print('set =' + str(n))

# graphable function of mu_y
def predicted_mean(x1, x2):
    mu_z = mu_w[0] * x1 + mu_w[1] * x2
    var_z = sigma_w[0][0] * x1 ** 2 + 2 * sigma_w[0][1] * x1 * x2 + sigma_w[1][1] * x2 ** 2
    t = np.sqrt(1 + lamb ** 2 * var_z)
    mu_y = sigmoid(mu_z / t)
    # print('mu_y')
    # print(mu_y)
    return mu_y

# graphable function of var_y
def predicted_variance(x1, x2):
    mu_z = mu_w[0] * x1 + mu_w[1] * x2
    var_z = sigma_w[0][0] * x1 ** 2 + 2 * sigma_w[0][1] * x1 * x2 + sigma_w[1][1] * x2 ** 2
    t = np.sqrt(1 + lamb ** 2 * var_z)
    var_y = mu_y * (1 - mu_y) * (1 - (1 / t))
    print('var_y')
    print(var_y)
    return var_y

# plots
x1, x2 = np.linspace(-3, 3, 100), np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(x1, x2)

plt.contour(X1, X2, predicted_mean(X1, X2), [0.2, 0.4, 0.6, 0.8], colors='black')
plt.show()

plt.contour(X1, X2, predicted_variance(X1, X2), [0.0000001, 0.0000005], colors='black')
plt.show()