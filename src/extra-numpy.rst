Other important NumPy commands
=================================

Python is not a pre-requsite of this course so only a limited amount of NumPy has been covered.
This section summerizes the many of the other important features of NumPy for those who are interested.

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
...     print(element)
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
^^^^^^^^^

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
>>> a = np.array([[1,2,3],[4,5,np.nan],[7,8,9]])
>>> a
array([[  1.,   2.,   3.],
       [  4.,   5.,  nan],
       [  7.,   8.,   9.]])
       
>>> columnMean = np.nanmean(a,axis=0)
>>> columnMean
array([ 4.,  5.,  6.])
>>> rowMean = np.nanmean(a,axis=1)
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

There are a number of convenience functions to help create matrices

.. tip:: 

   >>> np.ones((3,2))
   >>> np.zeros((3,2))
   >>> np.eye(3)
   >>> np.diag([1,2,3])
   >>> np.fromfunction(lambda i, j: (i-2)**2+(j-2)**2, (5,5))

Getting more comfortable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These are some of the things that will become second nature to you once you get a little more comfortable with NumPy

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
