import gurobipy as gp
from gurobipy import GRB
import numpy as np
from pca_utils import disP

def optimize(A, dA, nc, k, epsilon, maxiter):
    n, p = A.shape
    iteration = 0
    Bindices = []
    Bvalue = 0

    m = gp.Model()
    s = m.addVars(range(p), vtype=GRB.BINARY, name="s")
    m.setObjective(sum(dA[i] * s[i] for i in range(p)), GRB.MAXIMIZE)
    m.addConstr(s.sum('*') <= k)

    def addCut(m, where):
        nonlocal iteration, Bindices, Bvalue
        if where == GRB.Callback.MIPSOL:
            if iteration < maxiter:
                sln = m.cbGetSolution(s)
                sn = np.zeros(p)
                indices = []
                for i in range(p):
                    sn[i] = sln[i]
                    if sn[i] > 0.5:
                        indices.append(i)
                sn = sn.astype(int)
                excess, objval = disP(A[:, indices], min([nc, sum(sn)]))
                if Bvalue <= objval:
                    Bvalue = objval
                    Bindices = indices
                violation = sum(excess) > epsilon
                if violation:
                    m.cbLazy(gp.quicksum(s[i] for i in indices) <= sum(sn) - 1)
                iteration += 1

    m.Params.lazyConstraints = 1
    m.optimize(addCut)
    return Bindices, Bvalue, iteration 