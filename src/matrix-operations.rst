.. probability lecture

First Unit
=============================


A probability distribution is a mathematical formalization that describes a 
particular type of random process. 


Properties of Distributions
-----------------------------

Probability Distributions are classified into two categories:

* **discrete** -- producing outcomes that can be mapped to the *integers* (such as 1, 2, ...) 

* **continuous** -- producing *real-valued* outcomes (such as 3.14... or 2.71...)

**Discrete distributions** are specified using 
**probability mass functions** 
often indicated as :math:`Pr(X=x)` 
while **continuous distributions** 
are specified using **probability density functions**
often indicated as :math:`f(X=x)`.

**Discrete distributions** specify probabilities of observing outcome :math:`x`
from a **discrete** random variable :math:`X` directly, 
while **continuous distributions** specify 
the behavior of realizations :math:`x` of a **continuous** random variable :math:`X`
in a retaliative rather than absolute manner.
For example, 
if :math:`f(X=x_1) = 2f(X=x_2)` then in the long-run 
:math:`x_2` will occur *twice as frequently* as :math:`x_1`.

Regardless of whether or not a 
random variable :math:`X` is discete or continuous,
if it is distributed according to a distribution named :math:`XYZ` with 
parameters :math:`\alpha` and :math:`\beta`, and so on, 
then we write 

.. math::
   X \sim XYZ(\alpha, \beta, ...)

and if 
a collection of :math:`n` random variables :math:`X_i, \; i=1, 2, \cdots n`
are **identically and independently distributed (i.i.d)**
---i.e., the random variables have *the same distribution*
and the realization of one *does not influence* the
realization of another--- then we write

.. math::
   X_i \overset{\small i.i.d.}{\sim} XYZ(\alpha, \beta, ...), \; i=1,2,\cdots n

..


.. note::

   **CLASS DISCUSSION**
   
   Let's say that I polled all first graders in the state of
   colorado and asked the question do you like/dislike your teacher.
   The answers are discrete values and the distribution of those
   answers could be modeled with a Bernoulli model. What are some other examples?

Check out this `khan academy video on the Bernoulli distribution
<https://www.khanacademy.org/math/statistics-probability/sampling-distributions-library/sample-proportions/v/mean-and-variance-of-bernoulli-distribution-example>`_
if you need some further intuition about Bernoulli distributions.

Breakout session
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Okay we are finished with unit-1

  * :download:`The first breakout session <breakout-1.ipynb>`
