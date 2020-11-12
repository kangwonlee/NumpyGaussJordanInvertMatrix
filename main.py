import typing
import pprint


Scalar = typing.Union[int, float]
Row = typing.Union[typing.List[Scalar], typing.Tuple[Scalar]]
Matrix = typing.Union[typing.List[Row], typing.Tuple[Row]]


def get_zero(n:int) -> Matrix:
    return [
        [0] * n for i in range(n)
    ]


def get_identity(n:int) -> Matrix:
    result = get_zero(n)
    for i in range(n):
        result[i][i] = 1

    return result


def augment_mats(A:Matrix, B:Matrix):
    assert len(A) == len(B)
    return [row_A + row_B for row_A, row_B in zip(A, B)]


def gauss_jordan(A:Matrix) -> Matrix:
    AX = augment_mats(A, get_identity(len(A)))
    
    # pivot loop
    for p in range(len(AX)):
        print(f"p = {p}")
        pprint.pprint(AX, width=50)

        one_over_pivot = 1.0 / AX[p][p]

        # normalize a row with one_over_pivot
        for j in range(len(AX[p])):
            AX[p][j] *= one_over_pivot

        print(f"p = {p} after normalization")
        pprint.pprint(AX, width=50)

        # row loop
        for i in range(len(AX)):
            if i != p:
                # row operation
                multiplier = - AX[i][p]

                # column loop
                for j in range(0, len(AX[p])):
                    AX[i][j] += multiplier * AX[p][j]

            print(f"p = {p} after i = {i}")
            pprint.pprint(AX, width=50)
            input("=== Press Enter ===")

    return [row[len(A):] for row in AX]


def mat_mul(A:Matrix, B:Matrix) -> Matrix:
  result = []

  for i, row in enumerate(A):
    new_row = [0] * len(B[0])
    for j in range(len(B[0])):
      for k, col in enumerate(B):
        new_row[j] += row[k] * col[j]

    result.append(new_row)

  return result


def main():
  A33_list = [
    [1, 0, 1],
    [0, 2, 1],
    [1, 1, 1],
  ]

  mat_A33_inv_GJ = gauss_jordan(A33_list)

  print("invert matrix B")
  pprint.pprint(mat_A33_inv_GJ, width=20)

  print("A B == indentical")
  pprint.pprint(mat_mul(A33_list, mat_A33_inv_GJ), width=20)


if "__main__" == __name__:
  main()
