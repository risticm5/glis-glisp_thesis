#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from pyswarm import pso
from glis.solvers import GLISp

#savefigs = False
savefigs = True

# implementation GLISp on statical dmps, we want to optimize parameter tau and delta_t
lb = np.array([0.0001, 0.0001])
ub = np.array([0.5, 0.5])
# function fun is unknown now
# and we don't know optimum/optimizer 

max_evals = 25
comparetol = 1e-4
n_initial_random = 10

max_prefs = max_evals - 1

####################################################################################
# Define synthetic preference function mapping (x1,x2) to {-1,0,1}
def pref_fun(x1, x2):
    #x1 is new point, and x2 is the current optimal   
    print("Compare dmps trajectory with 1) %s parameters with 2) %s parameters" % (x1, x2))
    print("Give the answer -1 , 0 , 1 if first better, equal, second better")
    input_var = input("Please enter some value: ")
    pref= int(input_var)
    return pref

##########################################

key = 10
np.random.seed(key)  # rng default for reproducibility
####################################################################################
print("Solve the problem by feeding the  preference expression step directly into the GLISp solver")
# Solve global optimization problem
prob1 = GLISp(bounds=(lb, ub), n_initial_random=n_initial_random)
xopt1 = prob1.solve(pref_fun, max_prefs)
X1 = np.array(prob1.X)
#fbest_seq1 = list(map(fun, X1[prob1.ibest_seq]))
##########################################

# Plot
print("Optimal parameters are tau =  %s and delta_t =  %s " % (xopt1[0], xopt1[1]))
print("Optimization finished. Draw the plot")

plt.plot(np.arange(0, max_evals), prob1.ibest_seq, color=[0.8500, 0.3250, 0.0980])
plt.ylim(0, max_evals)
plt.title("best iteration index") 
plt.xlabel("queries")
plt.grid()

if savefigs:
    plt.savefig("glisp_exp1_1.png", dpi=300)
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
ax[0].plot(np.arange(0, max_evals), X1[prob1.ibest_seq,0], color=[0.8500, 0.3250, 0.0980])
ax[0].set_ylim(lb[0], ub[0])
ax[0].set_title("optimal tau value through iterations")
ax[0].set_xlabel("queries")
ax[0].grid()

ax[1].plot(np.arange(0, max_evals), X1[prob1.ibest_seq,1], color=[0.8500, 0.3250, 0.0980])
ax[1].set_ylim(lb[1], ub[1])
ax[1].set_title("optimal delta_t value through iterations")
ax[1].set_xlabel("queries")
ax[1].grid()

if savefigs:
    plt.savefig("glisp_exp1_2.png", dpi=300)
plt.show()

"""
np.random.seed(key)
####################################################################################
print("Solve the problem incrementally (i.e., provide the preference at each iteration)")
# solve same problem, but incrementally
prob2 = GLISp(bounds=(lb, ub), n_initial_random=n_initial_random)
xbest2, x2 = prob2.initialize()  # get first two random samples
for k in range(max_prefs):
    pref = pref_fun(x2, xbest2)  # evaluate preference
    x2 = prob2.update(pref)
    xbest2 = prob2.xbest
X2 = np.array(prob2.X[:-1])
xopt2 = xbest2
##########################################
# assert np.linalg.norm(X1-X2)==0.0 and np.all(xopt1==xopt2)
print("Optimal parameters are tau =  %s and delta_t =  %s " % (xopt2[0], xopt2[1]))
"""