.. linear algebra

Matrix Decomposition
=============================

The idea of **Matrix decomposition** also known as **matrix factorization** is that 

more more


Eigendecomposition
------------------------

Let :math:`A` be an :math:`n \times n` matrix and :math:`\mathbf{x}` be an :math:`n \times 1` nonzero vector. An **eigenvalue** of :math:`A` is a number :math:`\lambda` such that

.. math::

   A \boldsymbol{x} = \lambda \boldsymbol{x}


A vector :math:`\mathbf{x}` satisfying this equation is called an **eigenvector** associated with :math:`\lambda`


>>> a = np.diag((1, 2, 3))
>>> a
array([[1, 0, 0],
       [0, 2, 0],
       [0, 0, 3]])
>>> w,v = np.linalg.eig(a)
>>> w;v
array([ 1.,  2.,  3.])
array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.]])

Eigenvectors and eigenvalues will play a huge roll in matrix methods later in the course (PCA, SVD, NMF).


