.. probability lecture

Matrix operations
=============================

Learning objectives:

  1. Understand the dimensional requirements for matrix multiplication
  2. Understand and be able to execute **elementwise** arithmetric operators in NumPy
  3. Be familiar with the commonly encountered matrix operations: slicing, masking and concatenation
  4. Be able to recall the output of vector-vector, matrix-vector, and matrix-matrix products
  5. Transpose

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

Dimensional requirements for matrix multiplication
----------------------------------------------------

.. important:: In order for the matrix product (:math:`A \times B`) to
               exist, the number of columns in :math:`A` must equal
               the number of rows in :math:`B`.


Basic properties of matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is convention to represent vectors as column matrices.

A **column matrix** in NumPy.

.. math::
    
    \mathbf{x} =
    \begin{pmatrix}
    3  \\
    4  \\
    5  \\
    6  
    \end{pmatrix}

>>> x = np.array([[3,4,5,6]]).T

.. note:: Again notice the pair of double brackets

A **row matrix** in NumPy.

.. math::

    \mathbf{x}^{T} =
    \begin{pmatrix}
    3 & 4 & 5 & 6
    \end{pmatrix}

>>> x = np.array([[3,4,5,6]])

Since this is linear algebra essentials with the goal of preparing you
for a learning experience in data science, lets introduce a running
example that will help ground much of the notation and concepts.

Machine learning can be roughly split into two types of learning problems.

   * **Supervised learning** - learn a mapping from inputs :math:`\mathbf{X}` to outputs :math:`y`
   * **Unsupervised learning** - given only :math:`\mathbf{X}`, learn interesting patterns in :math:`\mathbf{X}`

An example of this is to predict housing prices, which would be a
continuous :math:`\mathbf{y}`.  When we use a feature matrix
:math:`\mathbf{X}` to predict :math:`\mathbf{y}` it is an example of
**supervised learning**.  Our feature matrix would be a number of
column vectors :math:`\mathbf{x}` horizontally stacked together to
form :math:`\mathbf{X}`.  Examples of these features might be: the
square footage, the distance from public transport, the average
quality of schools and more.

If we wanted to discover patterns in :math:`\mathbf{X}` then we could take an
unsupervised approach such as clustering, which would be an example of
**unsupervised learning**.

We may access the individual elements of :math:`\mathbf{X}` through **indexing**

.. math::

     \mathbf{X} =
    \begin{pmatrix}
     X_{1,1} & X_{1,2} \\
     X_{2,1} & X_{2,2} \\
    \end{pmatrix}

In the above matrix we show how to explicitly access any element in
:math:`\mathbf{X}`.  It is also quite common to refer to a generic
element in :math:`\mathbf{X}` as :math:`\mathbf{X}_{i,j}`, where if
you picked up on the pattern indexing occurs with

    
This has already been stated once.  But since it is important lets say it a different way. 
    
.. note:: In order to multiply two matrices, they must be
          **conformable** such that the number of columns of the first
          matrix must be the same as the number of rows of the second
          matrix.


When we say multiply two matrices it does not mean multiply in the sense that you might think.
The **matrix product** of two matrices is another matrix.

If we have two vectors :math:`\mathbf{x}` and :math:`\mathbf{y}` of the same length :math:`(n)`, then the **dot product** is give by

.. math:: 

   \mathbf{x} \cdot \mathbf{y} = x_1y_1 + x_2y_2 + \cdots + x_ny_n$$

Basic operations
--------------------

Arithmetic operators in NumPy work **elementwise**.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

>>> a = np.array([3,4,5])
>>> b = np.ones(3)
>>> a - b
array([ 2.,  3.,  4.])

Something that can be tricky for people familar with other programming languages is that the * operator
**does not** carry out a matrix product.  This is done with the 
`dot <http://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html>`_ function.

>>> a = np.array([[1,2],[3,4]])
>>> b = np.array([[1,2],[3,4]])
>>> a
array([[1, 2],
       [3, 4]])
>>> b
array([[1, 2],
       [3, 4]])
>>> a * b
array([[ 1,  4],
       [ 9, 16]])
>>> np.dot(a,b)
array([[ 7, 10],
       [15, 22]])

Special addition and multiplication operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

>>> a = np.zeros((2,2),dtype='float')
>>> a += 5
>>> a
array([[ 5.,  5.],
       [ 5.,  5.]])
>>> a *= 5
>>> a
array([[ 25.,  25.],
       [ 25.,  25.]])
>>> a + a
array([[ 50.,  50.],
       [ 50.,  50.]])

Concatenation
^^^^^^^^^^^^^^^^^^^

>>> a = np.array([1,2,3])
>>> b = np.array([4,5,6])
>>> c = np.array([7,8,9])
>>> np.hstack([a,b,c])
array([1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> np.vstack([a,b,c])
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])

Sorting arrays
^^^^^^^^^^^^^^

>>> x.sort()
>>> x
array([0, 0, 1, 3, 4, 5])
>>> x = np.array(([1,3,4,0,0,5]))
array([3, 4, 0, 1, 2, 5])
>>> np.argsort(x)
array([3, 4, 0, 1, 2, 5])

Common math functions
^^^^^^^^^^^^^^^^^^^^^

>>> x = np.arange(1,5)
>>> np.sqrt(x) * np.pi 
array([ 3.14159265,  4.44288294,  5.44139809,  6.28318531])
>>> 2**4
16
>>> np.power(2,4)
16
>>> np.log(np.e)
1.0
>>> x = np.arange(5)
>>> x.max() - x.min()
4

Basic operations excercise
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: Exercise

   In the following table we have expression values for 5 genes at 4 time points.
   These are completely made up data.  Although, some of the questions can be 
   easily answered by looking at the data, microarray data generally come in much 
   larger tables and if you can figure it out here the same code will work for an 
   entire gene chip.  

   +------------+----------+----------+---------+----------+
   | Gene name  | 4h       | 12h      | 24h     | 48h      |
   +============+==========+==========+=========+==========+
   | A2M        | 0.12     | 0.08     | 0.06    | 0.02     |
   +------------+----------+----------+---------+----------+
   | FOS        | 0.01     | 0.07     | 0.11    | 0.09     |
   +------------+----------+----------+---------+----------+
   | BRCA2      | 0.03     | 0.04     | 0.04    | 0.02     |
   +------------+----------+----------+---------+----------+
   | CPOX       | 0.05     | 0.09     | 0.11    | 0.14     |
   +------------+----------+----------+---------+----------+

   1. create a single array for the data (4x4)
   2. find the mean expression value *per gene*
   3. find the mean expression value *per time point*
   4. which gene has the maximum mean expression value?
   5. sort the gene names by the max expression value

.. tip:: 

   >>> geneList = np.array(["A2M", "FOS", "BRCA2","CPOX"])
   >>> values0  = np.array([0.12,0.08,0.06,0.02])
   >>> values1  = np.array([0.01,0.07,0.11,0.09])
   >>> values2  = np.array([0.03,0.04,0.04,0.02])
   >>> values3  = np.array([0.05,0.09,0.11,0.14])

Indexing, slicing and more
-----------------------------

1D arrays can be indexed in the same way a Python list can.

>>> a = np.arange(10)
>>> a[2:4]
array([2, 3])
>>> a[:10:2]
array([0, 2, 4, 6, 8])
>>> a[::-1]
array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

Multidimensional arrays can have one index per axis

>>> x = np.arange(12).reshape(3,4)
>>> x
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> x[2,3]
11
>>> x[:,1]                       # everything in the second row
array([1, 5, 9])
>>> x[1,:]                       # everything in the second column
array([4, 5, 6, 7])
>>> x[1:3,:]                     # second and third rows
array([[ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])

Where
^^^^^

>>> a = np.array([1,1,1,2,2,2,3,3,3])
>>> a[a>1]
array([2, 2, 2, 3, 3, 3])
>>> a[a==3]
array([3, 3, 3])
>>> np.where(a<3)
(array([0, 1, 2, 3, 4, 5]),)
>>> np.where(a<3)[0]
array([0, 1, 2, 3, 4, 5])
>>> np.where(a>9)
(array([], dtype=int64),)

Printing
^^^^^^^^

>>> for row in x:
...     print row
... 
[0 1 2 3]
[4 5 6 7]
[ 8  9 10 11]

>>> for element in x.flat:
...     print element
... 
0
1
2
3
4
5
6
7
8
9
10
11

Copying
^^^^^^^

>>> a = np.array(['a','b','c'])
>>> b = a
>>> b[1] = 'z'
>>> a
array(['a', 'z', 'c'], 
      dtype='|S1')

>>> a = np.array(['a','b','c'])
>>> b = a.copy()
>>> b[1] = 'z'
>>> a
array(['a', 'b', 'c'], 
      dtype='|S1')

Missing data
^^^^^^^^^^^^

>>> import numpy as np
>>> from scipy.stats import nanmean 
>>> a = np.array([[1,2,3],[4,5,np.nan],[7,8,9]])
>>> a
array([[  1.,   2.,   3.],
       [  4.,   5.,  nan],
       [  7.,   8.,   9.]])
>>> columnMean = nanmean(a,axis=0)
>>> columnMean
array([ 4.,  5.,  6.])
>>> rowMean = nanmean(a,axis=1)
>>> rowMean
array([ 2. ,  4.5,  8. ])

Generating random numbers
^^^^^^^^^^^^^^^^^^^^^^^^^

>>> np.random.randint(0,10,5)      # random integers from a closed interval
array([2, 8, 3, 7, 8])
>>> np.random.normal(0,1,5)        # random numbers from a Gaussian
array([ 1.44660159, -0.35625249, -2.09994545,  0.7626487 ,  0.36353648])
>>> np.random.uniform(0,2,5)       # random numbers from a uniform distribution
array([ 0.07477679,  0.36409135,  1.42847035,  1.61242304,  0.54228665])

There are many other useful functions in `random <http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.html>`_

Convenience functions
^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: Exercise

   >>> print np.ones((3,2)), '\n'
   >>> print np.zeros((3,2)), '\n'
   >>> print np.eye(3), '\n'
   >>> print np.diag([1,2,3]), '\n'
   >>> print np.fromfunction(lambda i, j: (i-2)**2+(j-2)**2, (5,5))

Having fun
^^^^^^^^^^^^^^^

>>> n, nrows, ncols = 100, 10, 6
>>> xs = np.random.normal(n, 15, size=(nrows, ncols)).astype('int')
>>> xs
array([[ 84, 108,  96,  93,  82, 115],
[ 87,  70,  96, 132, 111, 108],
[ 96,  85, 120,  72,  62,  66],
[112,  86,  98,  86,  74,  98],
[ 75,  91, 116, 105,  82, 122],
[ 95, 119,  84,  89,  93,  87],
[118, 113,  94,  89,  67, 107],
[120, 105,  85, 100, 131, 120],
[ 91, 137, 103,  94, 115,  92],
[ 73,  98,  81, 106, 128,  75]])

Index it with a list of integers

>>> print(xs[0, [1,2,4,5]])

Boolean indexing

>>> print(xs[xs % 2 == 0])

What does this do?

>>> xs[xs % 2 == 0] = 0

Extracting lower triangular, diagonal and upper triangular matrices

>>> a = np.arange(16).reshape(4,4)
>>> print a, '\n'
>>> print np.tril(a, -1), '\n'
>>> print np.diag(np.diag(a)), '\n'
>>> print np.triu(a, 1)

