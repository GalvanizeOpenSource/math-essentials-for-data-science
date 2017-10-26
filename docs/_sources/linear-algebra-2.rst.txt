.. linear algebra, linear regression
   

Linear Algebra Part II
=======================================

These concepts build on what we learned in part I

Goals
^^^^^^^^^

Using NumPy build an intuition for basic linear algebra concepts:

  1. Norms and other special matrices
  2. Orthogonal Matrices

Norms and other special matrices
---------------------------------

.. figure:: vector.png
   :scale: 75%
   :align: center
   :alt: vector
   :figclass: align-center


The norm of a vector :math:`\mathbf{x}` is defined by

.. math::
   ||\mathbf{x}|| = \sqrt{x_1^2 + x_2^2 + \cdots + x_n^2}

>>> x = np.array([1,2,3,4])   
>>> print(np.sqrt(np.sum(x**2)))
>>> print(np.linalg.norm(x))
5.47722557505
5.47722557505

The norm squared of a vector is just the vector dot product with itself

.. math::

   ||x||^2 = x \cdot x

print(np.linalg.norm(x)**2)
print(np.dot(x,x))

The distance between two vectors is the norm of the difference.

.. math::

   d(x,y) = ||x-y||

>>> np.linalg.norm(x-y)
4.472

**Cosine Similarity** is the cosine of the angle between the two vectors give by

.. math::

   cos(\theta) = \frac{\mathbf{x} \cdot \mathbf{y}}{||\mathbf{x}|| \text{ } ||\mathbf{y}||}

>>> x = np.array([1,2,3,4])
>>> y = np.array([5,6,7,8])
>>> np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
0.96886393162696616

If both :math:`\mathbf{x}` and :math:`\mathbf{y}` are zero-centered, this calculation is the **correlation** between :math:`\mathbf{x}` and :math:`\mathbf{y}`.

>>> from scipy.stats import pearsonr
>>> x_centered = x - np.mean(x)
>>> y_centered = y - np.mean(y)
>>> r1 = np.dot(x_centered,y_centered)/(np.linalg.norm(x_centered)*np.linalg.norm(y_centered))
>>> r2 = pearsonr(x_centered,y_centered)
>>> print(r1,r2[0])
1.0 1.0

Special matrices
--------------------

Let :math:`X` be a matrix of dimension :math:`n \times k` and let :math:`Y` be a matrix
of dimension :math:`k \times p`, then the product :math:`XY` will be a matrix of
dimension :math:`n \times p` whose :math:`(i,j)^{th}` element is given by the dot
product of the :math:`i^{th}` row of :math:`X` and the :math:`j^{th}` column of :math:`Y`

.. math::
   \sum_{s=1}^k x_{is}y_{sj} = x_{i1}y_{1j} + \cdots + x_{ik}y_{kj}

**Orthogonal Matrices**

Let :math:`X` be an :math:`n \times n` matrix such than :math:`X^TX = I`, then :math:`X` is said to be orthogonal which implies that :math:`X^T=X^{-1}`

This is equivalent to saying that the columns of :math:`X` are all orthogonal to each other (and have unit length).

System of equations
---------------------

A system of equations of the form:

.. math::
   \begin{align*}
   a_{11}x_1 + \cdots + a_{1n}x_n &= b_1 \\
   \vdots \hspace{1in} \vdots \\
   a_{m1}x_1 + \cdots + a_{mn}x_n &= b_m
   \end{align*}

can be written as a matrix equation:

.. math::
   A\mathbf{x} = \mathbf{b}

and hence, has solution

.. math::
   \mathbf{x} = A^{-1}\mathbf{b}



       
Additional Properties of Matrices
------------------------------------

1. If :math:`X` and :math:`Y` are both :math:`n \times p` matrices,then

   .. math::
      X+Y = Y+X

2. If :math:`X`, :math:`Y`, and :math:`Z` are all :math:`n \times p` matrices, then

   .. math::
      X+(Y+Z) = (X+Y)+Z
   
3. If :math:`X`, :math:`Y`, and :math:`Z` are all conformable,then

   .. math::
      X(YZ) = (XY)Z
   
4. If :math:`X` is of dimension :math:`n \times k` and :math:`Y` and
:math:`Z` are of dimension :math:`k \times p`, then

   .. math::
      X(Y+Z) = XY + XZ

5. If :math:`X` is of dimension :math:`p \times n` and :math:`Y` and
:math:`Z` are of dimension :math:`k \times p`, then

   .. math::
      (Y+Z)X = YX + ZX
   
6. If :math:`a` and :math:`b` are real numbers, and :math:`X` is an :math:`n \times p` matrix, then

   .. math::
      (a+b)X = aX+bX
   
7. If :math:`a` is a real number, and :math:`X` and :math:`Y` are both :math:`n \times p` matrices,then

   .. math::
      a(X+Y) = aX+aY
   
8. If :math:`z` is a real number, and :math:`X` and :math:`Y` are conformable, then

   .. math::
      X(aY) = a(XY)

Another resource is this Jupyter notebook that review much of these materials

   * :download:`../notebooks/linear-algebra.ipynb`
  
