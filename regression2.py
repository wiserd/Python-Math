import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading Data
data = pd.read_csv('data4.csv',sep=r'\s*,\s*',
                           header=0, encoding='ascii', engine='python')
print(data.shape)
(237, 4)



# Coomputing X and Y


years = data['Year'].values
X = np.array(years)

for i in range(len(years)):
  X[i] = years[i]- 1980


#print(data.head())
print("ccccccccccccccccc" )
print(X )
print("aaaaaaaaaaaaaaaa" )


Y = data['CO2 PPM'].values
print("X=", X)
print("Y=", Y)

# Mean X and Y
mean_x = np.mean(X)
mean_y = np.mean(Y)
 
# Total number of values
n = len(X)

# Using the formula to calculate 'm' and 'c'
numer = 0
denom = 0
for i in range(n):
	numer += (X[i] - mean_x) * (Y[i] - mean_y)
	denom += (X[i] - mean_x) ** 2
	m = numer / denom
	c = mean_y - (m * mean_x)
 
# Printing coefficients
print("Coefficients")
print(m, c)

# Plotting Values and Regression Line
 
max_x = np.max(X) + 100
min_x = np.min(X) - 100
 
# Calculating line values x and y
x = np.linspace(min_x, max_x, 1000)
y = c + m * x
 
# Ploting Line
plt.plot(x, y, color='#58b970', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y, c='#ef5423', label='Scatter Plot')
 
plt.xlabel('Years from 1980')
plt.ylabel('CO2 in PPM')
plt.legend()
plt.show()


# Calculating Root Mean Squares Error
rmse = 0
for i in range(n):
    y_pred = c + m * X[i]
    rmse += (Y[i] - y_pred) ** 2
rmse = np.sqrt(rmse/n)
print("RMSE")
print(rmse)
# Calculating R2 Score
ss_tot = 0
ss_res = 0
for i in range(n):
    y_pred = c + m * X[i]
    ss_tot += (Y[i] - mean_y) ** 2
    ss_res += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_res/ss_tot)
print("R2 Score")
print(r2)