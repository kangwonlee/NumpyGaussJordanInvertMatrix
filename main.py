import numpy as np
import matplotlib.pyplot as plt


Matrix = np.ndarray


def gauss_jordan(A:Matrix) -> Matrix:
    assert 2 == len(A.shape), A.shape

    n = A.shape[0]
    m = A.shape[1]

    assert n == m, A.shape

    I = np.identity(n)

    AX = np.hstack([A, I])
    m = AX.shape[1]
    
    # pivot loop
    for p in range(len(AX)):
        print(f"p = {p}")
        print(AX)

        one_over_pivot = 1.0 / AX[p, p]

        # normalize a row with one_over_pivot
        AX[p, :] *= one_over_pivot

        print(f"p = {p} after normalization")
        print(AX)

        # row loop
        for i in range(len(AX)):
            if i != p:
                # row operation
                multiplier = - AX[i, p]

                # column operation
                AX[i, :] += multiplier * AX[p, :]

            print(f"p = {p} after i = {i}")
            print(AX)
            input("=== Press Enter ===")

    return np.hsplit(AX, 2)[-1]


def main():
  A33_list = [
    [1, 0, 1],
    [0, 2, 1],
    [1, 1, 1],
  ]

  A33_mat = np.array(A33_list)

  mat_A33_inv_GJ = gauss_jordan(A33_mat)

  print("invert matrix B")
  print(mat_A33_inv_GJ)

  print("A B == identity matrix")
  print(A33_mat @ mat_A33_inv_GJ)


if "__main__" == __name__:
  main()
