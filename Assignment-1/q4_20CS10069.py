from random import choice
import numpy as np
from matplotlib import pyplot
import cv2


def convolution_2D(A, B):
    m, n = A.shape
    p, q = B.shape

    C = np.array([[0] * (n + q - 1) for _ in range(m + p - 1)])
    for r in range(1, m+p):
        for s in range(1, n+q):
            for i in range(1, m+1):
                k = r + 1 - i
                if k > 0 and k <= p:
                    for j in range(1, n+1):
                        l = s + 1 - j
                        if l > 0 and l <= q:
                            C[r-1][s-1] += A[i-1][j-1] * B[k-1][l-1]
    return C


def main():

    ########------------------ PART A ------------------########

    A = np.array([[choice([0, 1]) for _ in range(6)] for _ in range(8)])
    B = np.array([[-1, 1], [1, -1]])

    print("A = \n", A)
    print("B = \n", B)
    C = convolution_2D(A, B)
    print("C = \n", C)

    ########------------------ PART B ------------------########
    B = [[1, -1]]
    D = np.array(B)

    imgs = ["0.jpg", "1.jpg", "9.jpg"]
    pos = 1

    for img in imgs:
        A = cv2.imread(img, 0)
        C = convolution_2D(A, D)

        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                if C[i][j] < 0:
                    C[i][j] = 0
                elif C[i][j] > 255:
                    C[i][j] = 255
        
        pyplot.subplot(3, 2, pos)
        pyplot.imshow(A, cmap='gray')
        pyplot.subplot(3, 2, pos+1)
        pyplot.imshow(C, cmap='gray')
        pos+=2

    pyplot.suptitle("Convolution of 2D images")
    pyplot.show()

if __name__ == "__main__":
    main()