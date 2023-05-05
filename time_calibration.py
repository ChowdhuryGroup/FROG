import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_image(filename: str):
    image = Image.open(filename)
    return np.array(image)


def gaussian_2d(point: tuple, amplitude, yo, sigma_x, sigma_y, offset):
    theta = 0.0
    x = point[0]
    y = point[1]
    xo = float(1106)
    yo = float(yo)
    a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (
        2 * sigma_y**2
    )
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (
        4 * sigma_y**2
    )
    c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (
        2 * sigma_y**2
    )
    g = offset + amplitude * np.exp(
        -(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2))
    )
    return g.ravel()


if __name__ == "__main__":
    file = open("time_calibration_data.tsv", "w")

    for i in range(60, 70):
        image = load_image("../fs_2023_04_25_Merge_" + str(i) + ".tiff")
        # Create x and y indices
        x = np.arange(image.shape[1])
        y = np.arange(image.shape[0])
        x, y = np.meshgrid(x, y)

        # create data
        data = image.ravel()

        # plot twoD_Gaussian data generated above
        plt.figure()
        plt.imshow(data.reshape(len(x), len(y)))
        plt.colorbar()

        # amplitude, yo, sigma_x, sigma_y, offset
        initial_guess = (4.0e4, 1000, 20, 50, 8.0e2)

        print("fitting")
        popt, pcov = opt.curve_fit(gaussian_2d, (x, y), data, p0=initial_guess)
        print("fitting complete")
        data_fitted = gaussian_2d((x, y), *popt)

        fig, ax = plt.subplots(1, 1)
        ax.imshow(
            data.reshape(len(x), len(y)),
            cmap=plt.cm.jet,
            origin="lower",
            extent=(x.min(), x.max(), y.min(), y.max()),
        )
        ax.contour(x, y, data_fitted.reshape(len(x), len(y)), 8, colors="w")
        plt.show()

        output_string = f"{i}\t{popt[1]}\n"
        print(output_string)
        file.write(output_string)
    file.close()
