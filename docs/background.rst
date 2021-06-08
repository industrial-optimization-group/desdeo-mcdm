Background concepts
===================

NAUTILUS
-----------------

In NAUTILUS, starting from the nadir point, a solution is obtained at each iteration which dominates the previous one. Although only the last solution will be Pareto optimal, the decision maker never looses sight of the Pareto optimal set, and the search is oriented so that (s)he progressively focusses on the preferred part of the Pareto optimal set. Each new solution is obtained by minimizing an achievement scalarizing function including preferences about desired improvements in objective function values.

The decision maker has two possibilities to provide her/his preferences:

1. The decision maker can rank the objectives according to the relative importance of improving each current objective value.

.. note::

	This ranking is not a global preference ranking of the objectives, but represents the local importance of improving each of the current objective values at that moment.

2. The decision maker can specify percentages reflecting how (s)he would like to improve the current objective values, by answering to the following question:

“Assuming you have one hundred points available, how would you distribute them among the current objective values so that the more points you allocate, the more improvement on the corresponding current objective value is desired?”

After each iteration round, the decision maker specifies whether (s)he wishes to continue with the previous preference information, or define a new one.

In addition to this, the decision maker can influence the solution finding process by taking a step back to previous iteration point. This enables the decision maker to provide new preferences and change the direction of solution seeking process. Furthermore, the decision maker can also take a half-step in case (s)he feels that a full step limits the reachable area of Pareto optimal set too much.

NAUTILUS is specially suitable for avoiding undesired anchoring effects, for example in negotiation support problems, or just as a means of finding an initial Pareto optimal solution for any interactive procedure.


NIMBUS
---------------

As its name suggests, NIMBUS (Nondifferentiable Interactive Multiobjective BUndle-based optimization System) is a multiobjective optimization system able to handle even non-differentiable functions. It will optimize (minimize or maximize) several functions simultaneously, creating a group of different solutions. One cannot say which one of them is the best, because the system cannot know the criteria affecting the 'goodness' of the desired solution. The user is the one that makes the decision.

Mathematically, all the generated solutions are 'equal', so it is important
that the user can influence the solution process. The user may want to choose which of the functions should be optimized most, the limits of the objectives, etc. In NIMBUS, this phase is called a 'classification'. 
Searching for the desired solution means finding the best compromise
between many different goals. If we want to get lower values for one function, we must be ready to accept the growth of another function. This is because the solutions produced by NIMBUS are Pareto optimal. This means that there is no possibility to achieve better solutions for some component of the problem without worsening some other component(s).


The Reference Point Method
--------------------------
In the Reference Point Method, the Decision Maker (DM) specifies desirable aspiration levels for objective functions. Vectors formed of these aspiration levels are then used to derive scalarizing functions having minimal values at weakly, properly or Pareto optimal solutions. It is important that reference points are intuitive and easy for the DM to specify, their consistency is not an essential requirement. Before the solution process starts, some information is given to the DM about the problem. If possible, the ideal objective vector and the (approximated) nadir objective vector are presented.

At each iteration, the DM is asked to give desired aspiration levels for the objective functions. Using this information to formulate a reference point, achievement function is minimized and a (weakly, properly or) Pareto optimal solution is obtained. This solution is then presented to the DM. In addition, k other (weakly, properly or) Pareto optimal solutions are calculated using perturbed reference points, where k is the number of objectives in the problem. The alternative solutions are also presented to the DM. If (s)he finds any of the k + 1 solutions satisfactory, the solution process is ended. Otherwise, the DM is asked to present a new reference point and the iteration described above is repeated.

The idea in perturbed reference points is that the DM gets better understanding of the possible solutions around the current solution. If the reference point is far from the Pareto optimal set, the DM gets a wider description of the Pareto optimal set and if the reference point is near the Pareto optimal set, then a finer description of the Pareto optimal set is given.

In this method, the DM has to specify aspiration levels and compare objective vectors. The DM is free to change her/his mind during the process and can direct the solution process without being forced to understand complicated concepts and their meaning. On the other hand, the method does not necessarily help the DM to find more satisfactory solutions.


NAUTILUS 2
----------
Similarly to NAUTILUS, starting from the nadir point, a solution is obtained at each iteration which dominates the previous one. Although only the last solution will be Pareto optimal, the Decision Maker (DM) never looses sight of the Pareto optimal set, and the search is oriented so that (s)he progressively focusses on the preferred part of the Pareto optimal set. Each new solution is obtained by minimizing an achievement scalarizing function including preferences about desired improvements in objective function values.

NAUTILUS 2 introduces a new preference handling technique which is easily understandable for the DM and allows the DM to conveniently control the solution process. Preferences are given as direction of improvement for objectives. In NAUTILUS 2, the DM has three ways to do this:

1. The DM sets the direction of improvement directly.

2. The DM defines the improvement ratio between two different objectives :math:`f_i` and :math:`f_j`. For example, if the DM wishes that the improvement of fi by one unit should be accompanied with the improvement of :math:`f_j` by :math:`θ_{ij}` units. Here, the DM selects an objective :math:`f_{i} (i=1,…,k)` and for each of the other objectives :math:`f_j` sets the value :math:`θ_{ij}`. Then, the direction of improvement is defined by

.. math::
	δ_i=1\ and\ δ_j=θ_{ij},\ j≠i.

3. As a generalization of the approach 2, the DM sets values of improvement ratios freely for some selected pairs of objective functions.

As with NAUTILUS, after each iteration round, the decision maker specifies whether (s)he wishes to continue with the previous preference information, or define a new one.

In addition to this, the decision maker can influence the solution finding process by taking a step back to the previous iteration point. This enables the decision maker to provide new preferences and change the direction of the solution seeking process. Furthermore, the decision maker can also take a half-step in case (s)he feels that a full step limits the reachable area of the Pareto optimal set too much.
