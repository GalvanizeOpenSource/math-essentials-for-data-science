Norms
=============================

Learning objectives

  1. transpose, dot products, inverse matrices
  2. matrix inversions and determinants 


transpose, dot products,identity matrices
-----------------------------------------

Basic properties of matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is convention to represent vectors as column matrices.  Think of
everything as a feature matrix a single :math:`x` is then a slice of
that matrix.


A **column matrix** in NumPy.

.. math::
    
    x =
    \begin{pmatrix}
    3  \\
    4  \\
    5  \\
    6  
    \end{pmatrix}

>>> x = np.array([[3,4,5,6]]).T

A **row matrix** in NumPy.

.. math::

    x =
    \begin{pmatrix}
    3 & 4 & 5 & 6
    \end{pmatrix}

>>> x = np.array([[3,4,5,6]])

General matrices like you saw :doc:`working with NumPy <matrix-operations>`.

.. math::

     A_{m,n} =
    \begin{pmatrix}
     a_{1,1} & a_{1,2} & \cdots & a_{1,n} \\
     a_{2,1} & a_{2,2} & \cdots & a_{2,n} \\
     \vdots  & \vdots  & \ddots & \vdots  \\
     a_{m,1} & a_{m,2} & \cdots & a_{m,n}
    \end{pmatrix}

    
.. note:: In order to multiply two matrices, they must be
          **conformable** such that the number of columns of the first
          matrix must be the same as the number of rows of the second
          matrix.


A :math:`1 \times N` dimensional vector :math:`x` 

.. math::

    x =
    \begin{pmatrix}
    x_{1}  \\
    x_{2}  \\
    \vdots \\
    x_{N}
    \end{pmatrix} 

and its transpose :math:`\mathbf{x}^{T} = (x_{1}, x_{2},\ldots,x_{N})` can be expressed in python as

>>> import numpy as np
>>> x = np.array([[1,2,3]]).T
>>> xt = x.T
>>> x.shape
(3, 1)
>>> xt.shape
(1, 3)

The transpose of a :math:`n \times p` matrix is a :math:`p \times n` matrix with rows and columns interchanged

.. math::
   
   X^T =
   \begin{bmatrix}
   x_{11} & x_{12} & \cdots & x_{1n} \\
   x_{21} & x_{22} & \cdots & x_{2n} \\
   \vdots & \vdots & \ddots & \vdots \\
   x_{p1} & x_{p2} & \cdots & x_{pn}
   \end{bmatrix}

Properties of a transpose
---------------------------

1. Let :math:`X` be an :math:`n \times p` matrix and :math:`a` a real number, then

   .. math::
      (cX)^T = cX^T

2. Let :math:`X` and :math:`Y` be :math:`n \times p` matrices, then

   .. math::
      (X \pm Y)^T = X^T \pm Y^T

3. Let :math:`X` be an :math:`n \times k` matrix and :math:`Y` be a :math:`k \times p` matrix, then

   .. math::
      (XY)^T = Y^TX^T

    
Dot products
----------------

>>> x = np.array([1,2,3,4])

Adding a constant to a vector adds the constant to each element

.. math::

   a + \mathbf{x} = [a + x_1, a + x_2, \ldots, a + x_n]

>>> print(x + 4)
[5 6 7 8]

Multiplying a vector by a constant multiplies each term by the constant

.. math::

   a \mathbf{x} = [ax_1, ax_2, \ldots, ax_n]

>>> print(x*4)
[ 4  8 12 16]

If we have two vectors :math:`\mathbf{x}` and :math:`\mathbf{y}`
of the same length :math:`n`, then the **dot product** is give by

.. math::
  \mathbf{x} \cdot \mathbf{y} = x_1 y_1 + x_2 y_2 + \cdots + x_ny_n

>>> y = np.array([4, 3, 2, 1])
>>> np.dot(x,y)
20

If :math:`\mathbf{x} \cdot \mathbf{y} = 0` then :math:`x` and :math:`y` are **orthogonal** (aligns with the intuitive notion of perpindicular)

>>> w = np.array([1, 2])
>>> v = np.array([-2, 1])
>>> np.dot(w,v)
0

If we have two vectors :math:`\mathbf{x}` and :math:`\mathbf{y}` of the
same length :math:`n`, then the **dot product** is give by matrix multiplication

.. math::

   \mathbf{x}^T \mathbf{y} =
   \begin{bmatrix} x_1& x_2 & \ldots & x_n \end{bmatrix}
   \begin{bmatrix}
   y_{1}\\
   y_{2}\\
   \vdots\\
   y_{n}
   \end{bmatrix}  =
   x_1y_1 + x_2y_2 + \cdots + x_ny_n


Matrix determinant
--------------------

The determinant of a 2-D array [[a, b], [c, d]] is ad - bc:

>>> a = np.array([[1, 2], [3, 4]])
>>> np.linalg.det(a)
-2.0

Matrix inverse
----------------

The inverse of a square :math:`n \times n` matrix :math:`X` is an :math:`n \times n` matrix :math:`X^{-1}` such that

.. math::
   X^{-1}X = XX^{-1} = I

Where :math:`I` is the identity matrix, an :math:`n \times n` diagonal matrix with 1's along the diagonal.

.. note:: If such a matrix exists, then :math:`X` is said to be
          **invertible** or **nonsingular** otherwise :math:`X` is
          said to be **noninvertible** or **singular**

>>> A = np.array([[-4,-2],[5,5]])
>>> A
array([[-4, -2],
       [ 5,  5]])
>>> invA = np.linalg.inv(A)
>>> invA
array([[-0.5, -0.2],
       [ 0.5,  0.4]])

>>> np.round(np.dot(A,invA))
array([[ 1.,  0.],
       [ 0.,  1.]])

Because :math:`AA^{-1} = A^{-1}A = I`.

       
Properties of Inverse
^^^^^^^^^^^^^^^^^^^^^^

1. If :math:`X` is invertible, then :math:`X^{-1}` is invertible and

   .. math::
      (X^{-1})^{-1} = X
   
2. If :math:`X` and :math:`Y` are both :math:`n \times n` invertible
   matrices, then :math:`XY` is invertible and

   .. math::
      (XY)^{-1} = Y^{-1}X^{-1}
   
3. If :math:`X` is invertible, then :math:`X^T` is invertible and

   .. math::
      (X^T)^{-1} = (X^{-1})^T

       


