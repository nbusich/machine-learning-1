import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x = np.genfromtxt('SAT.csv', delimiter=',')
n, bins, patches = plt.hist(x, 20, facecolor = 'blue', alpha=0.7)
plt.title("Probability Density Function of SAT SCORE")
plt.show()
print("SAT Avg", np.mean(x))

y = np.genfromtxt('student_age.csv', delimiter=',')
n, bins, patches = plt.hist(y, 20, facecolor = 'blue', alpha=0.7)
plt.title("Probability Density Function of STUDENT AGE")
plt.show()
print("AGE Avg", np.mean(y))

z = np.genfromtxt('lunch_wait_time.csv', delimiter=',')
n, bins, patches = plt.hist(z, 20, facecolor = 'blue', alpha=0.7)
plt.title("Probability Density Function of LUNCH WAIT")
plt.show()
print("WAIT Avg", np.mean(z))

# Inverse CDF to find the number required for percentage
print(norm.ppf(0.95, 1000, 200))

# Regular CDF men
a = norm.cdf(67, 70, 3)
print("This amount of men are taller than 5'7: ", 1-a)

# Regular CDF women
b = norm.cdf(67, 64.5, 2.5)
print("This amount of women are shorter than 5'7: ", b)

# Generating CDF plots
points = np.array([[], []])

x1 = np.linspace(57, 80)
x2 = np.linspace(55, 75)

# Women Curve
pdf_w = norm.pdf(x2, 64.5, 2.5)
# Men Curve
pdf_m = norm.pdf(x1, 70, 3)

plt.plot(x1, pdf_m)
plt.plot(x2, pdf_w)
plt.title("Probability Density Function of Men and Women")
plt.show()