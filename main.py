import numpy as np
import pickle
import string
import math
import dimod
from dimod import ConstrainedQuadraticModel, Integer
from dwave.system import LeapHybridCQMSampler

"""
print("number of variables: ", cqm.num_quadratic_variables())
print("variables:", cqm.variables)
print("constraints: ", cqm.constraints)

sampler = LeapHybridCQMSampler()
sampleset = sampler.sample_cqm(cqm)
result = sampleset.record
#energy = result.energy.min()
energy = result.energy[0]
sol = result.sample[0]
print("sol: ", sol) 
"""

def READ_from_QAPLIB(file):
    contents = open(file).read()
    m = [list(map(int,item.split())) for item in contents.split('\n') if len(item) > 0 ]
    n = m[0][0]
    n1 = n + 1
    return n, np.array(m[1:n1]),np.array(m[n1:])

instance = "test"
n, flows,distances = READ_from_QAPLIB(instance + '.dat')
n2 = n**2
#rou12 OPT = 235528 (6,5,11,9,2,8,3,1,12,7,4,10)
#Had12 OPT = 1652  (3,10,11,2,12,5,6,7,8,1,4,9)
if instance == "rou12":
    target = 235528
else:
    if instance == "had12":
        target = 1652
print("flows:")
print(flows)
print("distances: ") 
print(distances)

x = [[dimod.Binary(f'x_{i}_{j}') for j in range(n)] for i in range(n)]

cqm = ConstrainedQuadraticModel()

# Constraint that every location has exactly one facility assigned
for i in range(n):
    cqm.add_constraint(sum(x[i]) == 1) #, label=f'location_has_facility_{i}')

# Constraint that every factory has exactly one location assigned
for j in range(n):
    cqm.add_constraint(sum(x[i][j] for i in range(n)) == 1) #, label=f'facility_in_location_{i}')

#Objective function to minimize   
cqm.set_objective(sum(flows[i,j] * distances[k,l] * x[i][k] * x[j][l]
     for i in range(n) for j in range(n) for k in range(n) for l in range(n)))

#print("number of variables: ", cqm.num_quadratic_variables())
#print("variables:", cqm.variables)
#print("constraints: ", cqm.constraints)
#print("objective: ", cqm.objective)

print("calling CQM Solver...")
sampler = LeapHybridCQMSampler()
sampleset = sampler.sample_cqm(cqm)
#print("sampleset:", sampleset)
samples = sampleset.record
energy = samples.energy.min()
print("energy: ", energy)
for i in range(len(samples.energy)):
    result = samples.sample[i]
    #reify the solution as a permutation
    perm = np.zeros((n,), dtype=int)
    for i in range(n):
        for j in range(n):
            if result[(i*n)+j] == 1:
                 perm[i]=j
    #compute the cost (from solution as permutation)
    cost = 0
    for i in range(n):
        for j in range(n):
            cost += flows[i,j]*distances[int(perm[i]),int(perm[j])]
    print("solution: ", perm, "with cost (computed from permutation):", cost, "and energy:", samples.energy[i])




