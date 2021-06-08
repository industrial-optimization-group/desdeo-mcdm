Background concepts
===================

What is NAUTILUS?
-----------------

Most interactive methods developed for solving multiobjective
optimization problems sequentially generate Pareto optimal solutions
and the decision maker must always trade-off to get a new
solution. Instead, the family of interactive trade-off-free methods
called NAUTILUS starts from the worst possible objective values and
improves every objective function step by step according to the
preferences of the decision maker.

Recently, the NAUTILUS family has been presented as a general NAUTILUS framework consisting of several modules. This general software framework for the NAUTILUS family facilitates the implementation of all the NAUTILUS
methods and even other interactive approaches. It has been
designed following an object-oriented architecture and consists of several
software blocks designed to cover the NAUTILUS framework's different requirements independently. The implementation is available as open-source code, enhancing its wide applicability.


What is NIMBUS?
---------------

As its name suggests, NIMBUS (Nondifferentiable Interactive Multiobjective BUndle-based optimization System) is a multiobjective optimization system able to handle even non-differentiable functions. It will optimize (minimize or maximize) several functions simultaneously, creating a group of different solutions. One cannot say which one of them is the best, because the system cannot know the criteria affecting the 'goodness' of the desired solution. The user is the one that makes the decision.

Mathematically, all the generated solutions are 'equal', so it is important
that the user can influence the solution process. The user may want to choose which of the functions should be optimized most, the limits of the objectives, etc. In NIMBUS, this phase is called a 'classification'. 
Searching for the desired solution means finding the best compromise
between many different goals. If we want to get lower values for one function, we must be ready to accept the growth of another function. This is because the solutions produced by NIMBUS are Pareto optimal. This means that there is no possibility to achieve better solutions for some component of the problem without worsening some other component(s).
