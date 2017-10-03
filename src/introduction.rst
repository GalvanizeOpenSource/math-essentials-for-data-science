.. probability lecture

Thinking in terms of vectors and matrices
============================================

Learning objectives:

  1. Become familiar with linear algebra's basic data structures: **scalar**, **vector**, **matrix**, **tensor**
  2. Create, manipulate, and generally begin to get comfortable with NumPy arrays

So you may be asking why?
---------------------------

.. figure:: xkcd_ml_and_la.png
   :scale: 35%
   :align: center
   :alt: xkcd-1838
   :figclass: align-center

`https://xkcd.com/1838 <https://xkcd.com/1838>`_

One of the most important parts of the modeling process is model inference
     
Scalars, vectors, matrices and tensors
------------------------------------------

Without knowing anything about vectors or matrices there is already a
good chance that you have some intuition for these concepts. Think of
a spreadsheet with rows and columns.  Within a given cell there exists
some value---lets call it a `scalar
<https://en.wikipedia.org/wiki/Scalar_(mathematics)>`_; scalars are
the contents of vectors and matrices.  If we think of the idea of a
column and the elements contained therein we now have a basis for the
concept of a `vector
<https://en.wikipedia.org/wiki/Row_and_column_vectors>`_.  More
specifically, this is referred to as a **column vector**.  The
elements of a row are accordingly referred to as a **row vector**.

We collectively refer to the columns and rows as `matrix
<https://en.wikipedia.org/wiki/Matrix_(mathematics)>`_.

.. note::
    A matrix with :math:`m` rows and :math:`n` columns is a :math:`m \times n` matrix and we refer to :math:`m` and :math:`n` as **dimensions**.

If a matrix is a two dimensional representation of data then a `tensor
<https://en.wikipedia.org/wiki/Tensor>`_ is the generalization of that
representation to any number of dimensions.  Lets say we copied our
spreadsheet and created several new tabs then we are now working with a tensor.

+------------------+-----------------------------------+---------------------------------------------------+
| Machine Learning | Notation                          | Description                                       |
+==================+===================================+===================================================+
| **Scaler**       | :math:`x`                         | a single real number (ints, floats etc)           |
+------------------+-----------------------------------+---------------------------------------------------+
| **Vector**       | :math:`X` or :math:`X^{T}`        | a 1D array of numbers (real, binary, integer etc) |
+------------------+-----------------------------------+---------------------------------------------------+
| **Matrix**       | :math:`\textbf{X}_{(n \times p)}` | a 2D array of numbers                             |
+------------------+-----------------------------------+---------------------------------------------------+
| **Tensor**       | :math:`\hat{f}`                   | an array generalized to n dimensions              |
+------------------+-----------------------------------+---------------------------------------------------+

Matrices are also tensors.  If we were working with a :math:`4 \times
4` matrix it can be described as a tensor of rank 2.  The `rank
<https://en.wikipedia.org/wiki/Rank_(linear_algebra)>`_ is
the formal term for the number of dimensions.

.. admonition:: Questions

    1. So what are the dimensions of the following matrix

    .. math::

        x =
        \begin{pmatrix}
        0 & 0 & 1 & 0 \\
        1 & 2 & 0 & 1 \\
        1 & 0 & 0 & 1
        \end{pmatrix} 
 
    .. container:: toggle

        .. container:: header

            **ANSWER**

        The matrix dimensions are :math:`3 \times 4`

    |
	
    2. Given a spreadsheet that has 3 tabs and each tab has 10 rows with 5 columns how might we represent that data with a tensor?

    .. container:: toggle

        .. container:: header

            **ANSWER**

        The tensor would be of rank 3 and have the following dimensions :math:`10 \times 5 \times 3`
      
|

An introduction to NumPy and Arrays
-----------------------------------------

Sometimes we need to write concepts on paper or see them in action
through code before we can effectively strengthen our understanding.
We will be learning the through a widely used Python package called
`NumPy <numpy.scipy.org>`_ to help bring to life essentials of linear
algebra.

In order to get the most out of this resource and to ensure that you
can actively follow along it is easiest if you install a working
Python environment.

    :doc:`Python installation guide <install-python>`

.. important:: Familiarity with the Python language is not a
               prerequisite for this primer.  The included code blocks
               are minimal and you should be able to follow even
               without prior experience in Python.


	       
Once Python is installed you can start an interactive Python
environment by typing the command `ipython` into a terminal.  `NumPy
<numpy.scipy.org>`_ is the *de facto* standard for numerical computing
in Python and it comes installed as part of the Conda bundle.  It is
`highly optimized <http://www.scipy.org/PerformancePython>`_ and
extremely useful for working with matrices.  The standard matrix class
in NumPy is called an `array
<http://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html>`_.
We will first get comfortable working with arrays then we will ease
our way into the essential concepts of linear algebra.  NumPy will
provide you with a tool explore all concepts presented here.

The standard syntax for importing the package NumPy into a Python environment is

>>> import numpy as np

.. note:: Examples of code (like the import statement above) are line
          by line, where each line begins with `>>>`.  This means that
          you can copy the code that comes after the line indicator
          directly into your interpreter

Arrays
^^^^^^^^^

Python is an `object-oriented
<https://en.wikipedia.org/wiki/Object-oriented_programming>`_
programming language.  The main object in NumPy is the *homogeneous*,
*multidimensional* array.  An `array
<http://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html>`_
is our programmatic way to represent vectors and matrices.  An example
is a matrix :math:`x`

.. math::

    x =
    \begin{pmatrix}
    1 & 2 & 3  \\
    4 & 5 & 6  \\
    7 & 8 & 9
    \end{pmatrix} 
 
can be represented as

>>> import numpy as np
>>> x = np.array([[1,2,3],[4,5,6],[7,8,9]])
>>> x
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
>>> x.shape
(3, 3)

The array :math:`x` has 2 dimensions.  The number of dimensions is
referred to as **rank**.  The ndim is the same as the number of axes or the
length of the output of x.shape

>>> x.ndim
2

>>> x.size
9

Arrays are especially convenient because of built-in methods.

>>> x.sum(axis=0)
array([12, 15, 18])
>>> x.sum(axis=1)
array([ 6, 15, 24]) 

>>> x.mean(axis=0)
array([ 4.,  5.,  6.])
>>> x.mean(axis=1)
array([ 2.,  5.,  8.])

But arrays are also useful because they interact with other NumPy functions as 
well as being the main data structure in so many other Python packages. To make a sequence of numbers, 
similar to *range* in the Python standard library, we use 
`arange <http://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html>`_.

>>> np.arange(10)
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> np.arange(5,10)
array([5, 6, 7, 8, 9])
>>> np.arange(5,10,0.5)
array([ 5. ,  5.5,  6. ,  6.5,  7. ,  7.5,  8. ,  8.5,  9. ,  9.5])

Also we can recreate the first matrix by **reshaping** the output of arange.

>>> x = np.arange(1,10).reshape(3,3)
>>> x
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])

Another similar function to arange is `linspace <http://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html>`_
which fills a vector with evenly spaced variables for a specified interval.

>>> x = np.linspace(0,5,5)
>>> x
array([ 0.  ,  1.25,  2.5 ,  3.75,  5.  ])

As a reminder you may access the Python documentation at anytime from the command line using

.. code-block:: none

    ~$ pydoc numpy.linspace

Visualizing linspace...

.. plot:: linspace-example.py
   :include-source:

Arrays may be made of different types of data.

>>> x = np.array([1,2,3])
>>> x.dtype
dtype('int64')
>>> x = np.array([0.1,0.2,0.3])
>>> x
array([ 0.1,  0.2,  0.3])
>>> x.dtype
dtype('float64')
>>> x = np.array([1,2,3],dtype='float64')
>>> x.dtype
dtype('float64')

There are several convenience functions for making arrays that are worth mentioning:
    * `zeros <http://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html>`_
    * `ones <http://docs.scipy.org/doc/numpy/reference/generated/numpy.ones.html>`_

>>> x = np.zeros([3,4])
>>> x
array([[ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.]])
>>> x = np.ones([3,4])
>>> x
array([[ 1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.]])

.. admonition:: Exercise

    1. Create the following array (1 line)

    .. math::

        a =
        \begin{pmatrix}
        1       & 2      & \cdots & 10      \\
        11      & 12     & \cdots & 20      \\
        \vdots  & \ddots & \ddots & \vdots  \\
        91      & 92     & \cdots & 100 
        \end{pmatrix}

    .. container:: toggle

        .. container:: header

            **ANSWER**

	>>> import numpy as np
	>>> a = np.arange(1,101).reshape(10,10)

    |
	
    2. Use the array object to get the rank, number of elements, rows and columns


    .. container:: toggle

        .. container:: header

            **ANSWER**

        >>> print("Rank: {}\nSize: {}\nDimensions: {}".format(a.ndim,a.size,a.shape))
        Rank: 2
        Size: 100
        Dimensions: (10, 10)

    |
	
    3. Get the mean of the rows and columns

    .. container:: toggle

        .. container:: header

            **ANSWER**
       
        >>> print("Row means: {}".format(a.mean(axis=1)))
        Row means: [  5.5  15.5  25.5  35.5  45.5  55.5  65.5  75.5  85.5  95.5]

	>>> print("Column means: {}".format(a.mean(axis=0)))
        Column means: [ 46.  47.  48.  49.  50.  51.  52.  53.  54.  55.]

    |	
	
    4. How do you create a vector that has exactly 50 points and spans the range 11 to 23?

    .. container:: toggle

        .. container:: header

            **ANSWER**

        >>> b = np.linspace(11,23,50)

    |
	
    5. [extra] If you want a peak at whats to come see what happens when you do the following
       
        * np.log(a) 
        * np.cumsum(a)
        * np.power(a,2)
   
More resources
^^^^^^^^^^^^^^^^^^^^^^

   * `NumPy homepage <numpy.scipy.org>`_
   * `Official NumPy tutorial <http://scipy.org/NumPy_Tutorial>`_
   * `NumPy for MATLAB users <http://www.scipy.org/NumPy_for_Matlab_Users>`_
