Linear Algebra I
=============================


+----+----------------------------------------------------------------------------------------------------------------------------+
| **Learning Objectives**                                                                                                         |
+====+============================================================================================================================+
| 1  | Develop an intuition for **matrix transpose**                                                                              |
+----+----------------------------------------------------------------------------------------------------------------------------+
| 2  | Become familiar with the notion of a **determinant**                                                                       |
+----+----------------------------------------------------------------------------------------------------------------------------+
| 3  | Become familiar with the process of a **matrix inverse**                                                                   |
+----+----------------------------------------------------------------------------------------------------------------------------+


transposes, dot products, determinants, and inverses
-------------------------------------------------------

Quick reference
---------------------

Here we provide a summary the important commands that have already been introduced.

+-----------------------------------+-------------------------------------------------------------+
| NumPy command                     | Note                                                        |
+===================================+=============================================================+
| a.ndim                            | returns the num. of dimensions or the **rank**              |
+-----------------------------------+-------------------------------------------------------------+
| a.shape                           | returns the num. of rows and colums                         |
+-----------------------------------+-------------------------------------------------------------+
| a.size                            | returns the num. of rows and colums                         |
+-----------------------------------+-------------------------------------------------------------+
| arange(start,stop,step)           | returns a sequence vector                                   |
+-----------------------------------+-------------------------------------------------------------+
| linspace(start,stop,steps)        | returns a evenly spaced sequence in the specificed interval |
+-----------------------------------+-------------------------------------------------------------+
| dot(a,b)                          | matrix multiplication                                       |
+-----------------------------------+-------------------------------------------------------------+
| vstack([a,b])                     | stack arrays a and b vertically                             |
+-----------------------------------+-------------------------------------------------------------+
| hstack([a,b])                     | stack arrays a and b horizontally                           |
+-----------------------------------+-------------------------------------------------------------+
| where(a>x)                        | returns elements from an array depending on condition       |
+-----------------------------------+-------------------------------------------------------------+
| argsort(a)                        | returns the sorted indices of an input array                | 
+-----------------------------------+-------------------------------------------------------------+


Transposes
-------------

A **matrix transpose** is an operation that Takes an :math:`m \times
n` matrix and turns into an :math:`n \times m` matrix where the rows
of the original matrix are the columns in the transposed matrix, and
visa versa.

Recall that it is convention to represent vectors as column matrices.  

A **column matrix** 

.. math::
    
    x =
    \begin{pmatrix}
    3  \\
    4  \\
    5  \\
    6  
    \end{pmatrix}

and when written using NumPy is as follows.
    
>>> x = np.array([[3,4,5,6]]).T


The ``.T`` indicates the use of a **transpose**, a matrix operation that you have been using already.  A **row matrix** is then written as:

.. math::

    x =
    \begin{pmatrix}
    3 & 4 & 5 & 6
    \end{pmatrix}

>>> x = np.array([[3,4,5,6]])

Just to ensure you *really know* this...


.. admonition:: Questions

    1. Create a row vector and a column vector version of the numbers 1-5 and print the shape of each.

       **Extra** can you do it with ``arange``?

       
    .. container:: toggle

        .. container:: header

            **ANSWER**
	
        |
	
	You could write out 1-5, but here we show how to do it with ``arange`` and the array function ``.reshape``.
	    
        >>> column_vector = np.arange(1,6).reshape(5,1)
        >>> column_vector.shape
        (5, 1)
        >>> row_vector = np.arange(1,6).reshape(1,5)
        >>> row_vector.shape
        (1, 5)

|

The transpose of a :math:`n \times m` matrix is a :math:`m \times n` matrix with rows and columns interchanged
A transpose can be thought of as the mirror image of a matrix across the main diagonal.

.. math::
   
   X^T =
   \begin{bmatrix}
   x_{11} & x_{12} & \cdots & x_{1n} \\
   x_{21} & x_{22} & \cdots & x_{2n} \\
   \vdots & \vdots & \ddots & \vdots \\
   x_{p1} & x_{p2} & \cdots & x_{pn}
   \end{bmatrix}

Properties of a transpose
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Let :math:`X` be an :math:`n \times m` matrix and :math:`a` a real number, then

   .. math::
      (cX)^T = cX^T

>>> np.array_equal((X*a).T,(X.T)*a)
True
      
2. Let :math:`X` and :math:`Y` be :math:`n \times p` matrices, then

   .. math::
      (X \pm Y)^T = X^T \pm Y^T

3. Let :math:`X` be an :math:`n \times k` matrix and :math:`Y` be a :math:`k \times p` matrix, then

   .. math::
      (XY)^T = Y^TX^T
      
More on dot products
------------------------------------

Dot products are a concept that will come up over and over in machine
learning so just to be sure that you grasp it lets review 
and expand on the concept some.

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
of the same length :math:`n`, then the **dot product** is given by

.. math::
  \mathbf{x} \cdot \mathbf{y} = x_1 y_1 + x_2 y_2 + \cdots + x_ny_n

>>> y = np.array([4, 3, 2, 1])
>>> np.dot(x,y)
20

or more explicitly
>>> np.dot(np.array([[1,2,3,4]]), np.array([[4,3,2,1]]).T)
array([[20]])

One aspect of dot product that we have not mentioned is how dot
products (and vectors for that matter) can be thought of as lines in
geometric space.  If :math:`\mathbf{x} \cdot \mathbf{y} = 0` then
:math:`x` and :math:`y` are **orthogonal** (aligns with the intuitive
notion of perpindicular)

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

The determinant of a 2-D array is :math:`ad - bc`:

.. math::

    x =
    \begin{bmatrix}
    a & b \\
    c & d \\  
    \end{bmatrix}
 
`<https://en.wikipedia.org/wiki/Determinant>`_

>>> a = np.array([[1, 2], [3, 4]])
>>> np.linalg.det(a)
-2.0

The determinant is a useful value that can be computed for a **square
matrix**.  Just as the name implies a square matrix is any matrix with
an equal number of rows and columns.  Matrices are sometimes used as
the engines to describe processes.  Each step of the process may be
considered a transition or transformation and the determinant in these
cases serves as a scaling factor for the transformation.

`<https://en.wikipedia.org/wiki/Stochastic_matrix>`_

Matrix inverse
----------------

To talk about matrix inversion we need to first introduce the
**identity matrix**.  An identity matrix is a matrix that does not
change any vector when we multiply that vector by that matrix.  We
construct one of these matrices by setting all of the entries along
the main diagonal to 1, while leaving all of the other entries at
zero.

>>> np.eye(4)
array([[ 1.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.],
       [ 0.,  0.,  1.,  0.],
       [ 0.,  0.,  0.,  1.]])

The inverse of a square :math:`n \times n` matrix :math:`X` is an :math:`n \times n` matrix :math:`X^{-1}` such that

.. math::
   X^{-1}X = XX^{-1} = I

Where :math:`I` is the identity matrix.

.. important:: If such a matrix exists, then :math:`X` is said to be
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

When :math:`A^{-1}` exists, several different algorithms exist for
finding it in closed form.  The identify matrix is useful for solving
systems of linear equations as we will see in the next section.

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

       
