#!/usr/bin/env python
import math
import random
import sys
import time
import numpy as np


def parseLinearSystem(filename):
    """
    Parse the system of linear equations and return the matrices and vectors.

    Input:
        filename: The name of the input file.

    Output:
        A: an nxn numpy array representing the coefficient matrix.
        b: an nx1 numpy array representing the right-hand side of the equations (answer vector).
        x: an nx1 numpy array representing the initial guess for the solution.
        max_iter: the maximum number of iterations to perform.
        tol: the tolerance for the convergence criterion.
    """

    inputs = []
    try:
        with open(filename, "r") as f:
            for line in f:
                # Parses each line to ignore comments and new lines in our input file.
                if line.startswith('#') or line.startswith('\n'):
                    continue
                inputs.append(line.rstrip())

        # Parses the matrix and 2 vectors as a numpy array of floats. Eval ensures that each element is numeric.
        A = np.array(eval(inputs[0]), float)
        b = np.array(eval(inputs[1]), float)
        x = np.array(eval(inputs[2]), float)

        max_iter = int(inputs[3])
        tol = float(inputs[4])
        return A, b, x, max_iter, tol
    except FileNotFoundError:
        input("A file with the given filename cannot be found! \n"
              "Random inputs will be used instead! Press enter to continue. ")
        return randomJacobiInputs()
    except (NameError, TypeError):
        print("An exception has occurred, please ensure the file input follows the appropriate format")
    exit()


def randomJacobiInputs():
    """
    Generates random inputs for the Jacobi algorithm if a valid filename is not provided.
    The generated inputs will be valid (No 0's on the leading diagonal and strictly diagonally dominant).

    Output:
        A: an nxn numpy array representing the coefficient matrix.
        b: an nx1 numpy array representing the right-hand side of the equations (answer vector).
        x: an nx1 numpy array representing the initial guess for the solution.
        max_iter: the maximum number of iterations to perform.
        tol: the tolerance for the convergence criterion.
    """

    # Generates a random integer to act as the dimensions for A, b and x
    dimension = random.randint(5, 30)

    # Generate an nxn array of random integers between 1 and 30
    # This prevents the leading diagonal from having a 0 element.
    A = np.random.randint(1, 30, (dimension, dimension)).astype(float)

    # Ensures that the matrix is strictly diagonally dominant by setting the value of each
    # element on the leading diagonal to the sum of magnitudes of each element in the row.
    for i in range(0, len(A)):
        A[i][i] = sum(abs(A[i][j]) for j in range(0, len(A)))

    # Generate nx1 arrays for our initial guess x, and result b, with random integers between 0 and 30
    b = np.random.randint(30, size=dimension).astype(float)
    x = np.random.randint(30, size=dimension).astype(float)

    max_iter = 10000
    tol = 0

    return A, b, x, max_iter, tol


def verifyValidity(A):
    """
    Verify that the matrix A has no 0's on its leading diagonal

    Input:
        A: an nxn numpy array representing the coefficient matrix

    Output:
        A bool indicating whether the matrix has no 0s on its leading diagonal
    """
    for i in range(0, len(A)):
        if A[i][i] == 0:
            return False
    return True


def verifyShape(A, b, x):
    """
    Verify that the matrices and vectors have the correct shape (dimensions)

    Input:
        A: an nxn numpy array representing the coefficient matrix.
        b: an nx1 numpy array representing the right-hand side of the equations (answer vector).
        x: an nx1 numpy array representing the initial guess for the solution.

    Output:
        A boolean indicating whether the matrices and vectors have the required shape.
    """

    if A.shape[0] != A.shape[1]:
        print("Please ensure that matrix A is square (has the same number of rows as columns).")
        return False

    if b.shape[0] != A.shape[0] or x.shape[0] != A.shape[0]:
        print("Please ensure that the linear system is square (it has the same number of variables as equations).\n"
              "This error typically means the dimensions of b or x are inconsistent with the dimensions of A.")
        return False

    return True


def verifyConvergence(A):
    """
    Verify that the matrix A is strictly diagonally dominant

    Input:
        A: an nxn numpy array representing the coefficient matrix

    Output:
        A bool indicating whether the matrix A is strongly diagonally dominant.
    """

    for i in range(0, len(A)):
        diagonal_element = abs(A[i][i])

        # Calculates the sum of each element in row "i" except for the element on the leading diagonal.
        row_sum = sum(abs(A[i][j]) for j in range(0, len(A)) if j != i)
        if row_sum >= diagonal_element:
            return False
    return True


def acceptNonConvergence():
    """
    Used to handle the user's input when the matrix is not strictly diagonally dominant

    Output:
        A boolean indicating whether the user has accepted that their input may not produce a convergent answer.
    """

    print("The coefficient matrix A is not strictly diagonally dominant - "
          "the Jacobi algorithm is unlikely to converge.")
    user_continue = input("Would you like to continue anyway? (Y/N) ")

    if user_continue.upper() == "Y":
        return True
    elif user_continue.upper() == "N":
        exit()

    print("That command is not recognised, please input either Y/N")
    return False


def jacobi(A, b, x, max_iter, tol):
    """
    Solve the system of linear equations Ax = b using the Jacobi method.

    Input:
        A: an nxn numpy array representing the coefficient matrix.
        b: an nx1 numpy array representing the right-hand side of the equations (answer vector).
        x: an nx1 numpy array representing the initial guess for the solution.
        max_iter: the maximum number of iterations to perform.
        tol: the tolerance for the convergence criterion.

    Output:
        x or x_new: an nx1 numpy array representing the solution to the system of equations.
        num_iter: the number of iterations performed.
    """

    # Initialize the iteration counter.
    num_iter = 0

    while num_iter < max_iter:
        # Initialize the vector x_new with the same shape and elements as x.
        x_new = np.copy(x)
        # Loop through each row (indicated by i), and column (indicated by j) of A
        for i in range(0, len(A)):
            row_sum = 0
            for j in range(0, len(A)):
                # Perform a matrix multiplication between row i of A and the vector x
                # (excluding the leading diagonal, when i = j).
                if i != j:
                    row_sum = row_sum + (A[i][j] * x[j])
            # Use this sum to calculate a new estimate for each element in x.
            x_new[i] = (b[i] - row_sum) / A[i][i]
        num_iter += 1
        print("\nIteration number", num_iter)
        print("Old solutions for x:", x)
        print("New solutions for x:", x_new)
        # If every element in x has changed by less than the tolerance
        # We have reached a convergence condition, and return x_new.
        if all(abs(x[i] - x_new[i]) < tol for i in range(0, len(x))):
            return x_new

        # If any element of x_new is positive or negative infinity, then the algorithm has diverged.
        if any(abs(x_new) == math.inf):
            print("The Jacobi algorithm will not converge in this case")
            exit()

        # Otherwise, set x to our new value and iterate again.
        x = x_new
    return x


def validateSolution(A, b, x):
    """
    Compare the result of A multiplied by our x to our expected value of b, and display the difference between them.

    Input:
        A: an nxn numpy array representing the coefficient matrix.
        b: an nx1 numpy array representing the right-hand side of the equations (answer vector).
        x: an nx1 numpy array representing our calculated solution for the linear system.
    """

    print("\nProvided b:", b)
    #  Uses the numpy library to calculate the row wise dot product of A and x (Matrix multiplication).
    calculated_ans = np.dot(A, x)
    # Converts each element to a float to avoid scientific notation: 2.200000e1 -> 22.000000
    print("Calculated b:", [float(element) for element in calculated_ans])
    print("Difference between calculated and provided:", b - calculated_ans)


def main(argv):
    if len(argv) == 1:
        A, b, x0, max_iter, tol = randomJacobiInputs()
    else:
        A, b, x0, max_iter, tol = parseLinearSystem(sys.argv[1])

    if not verifyValidity(A):
        print("The coefficient matrix A has at least one 0 on its leading diagonal! \n"
              "The Jacobi method will not work!")
        return

    if not verifyShape(A, b, x0):
        return

    while not verifyConvergence(A):
        if acceptNonConvergence():
            break

    print("Matrix A:", A, "\nVector b:", b, "\nVector x:", x0,
          "\nMax iterations:", max_iter, "\nError tolerance:", tol)
    input("Press Enter to begin Jacobi Algorithm! ")

    start = time.time()
    ans = jacobi(A, b, x0, max_iter, tol)
    # Checks if we have returned a None object (occurs when the algorithm has diverged to infinity)
    if ans is None:
        exit()
    validateSolution(A, b, ans)
    print("\nCalculated solution, x =", ans)
    print("Time taken (seconds):", time.time() - start)


if __name__ == "__main__":
    main(sys.argv)
