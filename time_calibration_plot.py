import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt


data = np.loadtxt("time_calibration_data.tsv")


def linear(x, m, b):
    return m * x + b


popt, pcov = opt.curve_fit(linear, data[:, 0], data[:, 1], p0=(-1, 1000))

plt.scatter(data[:, 0], data[:, 1])
plt.plot(
    data[:, 0], linear(data[:, 0], *popt), label=f"Slope is {popt[0]:.2f} pixels/step"
)
plt.xlabel("Steps")
plt.ylabel("Pixel Position")
plt.legend()
plt.show()
