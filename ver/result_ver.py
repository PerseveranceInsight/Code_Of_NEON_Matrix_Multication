import numpy as np

if __name__ == '__main__':
    matrix_a = np.array([[0.0, 4.0, 8.0, 12.0],
                         [1.0, 5.0, 9.0, 13.0],
                         [2.0, 6.0, 10.0, 14.0],
                         [3.0, 7.0, 11.0, 15.0]])
    matrix_b = np.array([[0.0, -4.0, -8.0, -12.0],
                         [-1.0, -5.0, -9.0, -13.0],
                         [-2.0, -6.0, -10.0, -14.0],
                         [-3.0, -7.0, -11.0, -15.0]])
    matrix_r = np.matmul(matrix_a,matrix_b)
    print('matrix_a : {0}'.format(matrix_a))
    print('matrix_b : {0}'.format(matrix_b))
    print('matrix_r : {0}'.format(matrix_r))
