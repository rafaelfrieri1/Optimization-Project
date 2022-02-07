from pyomo.environ import *

CS = 10 # Magic number while creating the appropriate instances
CP = 40 # Magic number while creating the appropriate instances
CK = 200 # Magic number while creating the appropriate instances
rP = 5 # Magic number while creating the appropriate instances
rQ = 7 # Magic number while creating the appropriate instances
l = 0.5 # Magic number while creating the appropriate instances

ettaiP = [100, 80, 20, 45, 150, 35, 10, 250, 60, 300] # Magic numbers while creating the appropriate instances
ettajQ = [10, 150, 20, 140, 30, 130, 40, 120, 50, 110, 60, 100, 70, 90, 80] # Magic numbers while creating the appropriate instances
gammaiP = [12, 37, 80, 25, 97, 65, 73, 120, 10, 3] # Magic numbers while creating the appropriate instances
gammajQ = [1, 15, 2, 14, 3, 13, 4, 12, 5, 11, 6, 10, 7, 9, 8] # Magic numbers while creating the appropriate instances
dK = [200, 250, 100, 230, 1000, 400, 450, 700, 130, 1500] # Magic numbers while creating the appropriate instances

deltasi = [
  [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
  [50, 45, 40, 35, 30, 25, 20, 15, 10, 5],
  [7, 12, 3, 8, 17, 12, 2, 10, 1, 24],
  [33, 1, 100, 46, 32, 17, 35, 18, 14, 38],
  [112, 95, 30, 23, 1, 44, 74, 98, 30, 40]
] # Magic numbers while creating the appropriate instances

deltaij = [
  [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
  [75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5],
  [7, 12, 3, 8, 17, 12, 2, 10, 1, 24, 15, 4, 20, 11, 3],
  [33, 1, 100, 46, 32, 17, 35, 18, 14, 38, 55, 67, 83, 46, 20],
  [112, 95, 30, 23, 1, 44, 74, 98, 30, 40, 150, 93, 80, 101, 200],
  [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
  [30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2],
  [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
  [150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
  [23, 55, 12, 90, 250, 37, 40, 91, 15, 21, 44, 137, 45, 89, 123]
] # Magic numbers while creating the appropriate instances

deltaiip = [
  [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
  [75, 70, 65, 60, 55, 50, 45, 40, 35, 30],
  [7, 12, 3, 8, 17, 12, 2, 10, 1, 24],
  [33, 1, 100, 46, 32, 17, 35, 18, 14, 38],
  [112, 95, 30, 23, 1, 44, 74, 98, 30, 40],
  [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
  [30, 28, 26, 24, 22, 20, 18, 16, 14, 12],
  [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
  [150, 140, 130, 120, 110, 100, 90, 80, 70, 60],
  [23, 55, 12, 90, 250, 37, 40, 91, 15, 21]
] # Magic numbers while creating the appropriate instances

deltajk = [
  [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
  [75, 70, 65, 60, 55, 50, 45, 40, 35, 30],
  [7, 12, 3, 8, 17, 12, 2, 10, 1, 24],
  [33, 1, 100, 46, 32, 17, 35, 18, 14, 38],
  [112, 95, 30, 23, 1, 44, 74, 98, 30, 40],
  [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
  [30, 28, 26, 24, 22, 20, 18, 16, 14, 12],
  [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
  [150, 140, 130, 120, 110, 100, 90, 80, 70, 60],
  [23, 55, 12, 90, 250, 37, 40, 91, 15, 21],
  [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
  [20, 18, 16, 14, 12, 10, 8, 6, 4, 2],
  [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
  [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
  [97, 10, 34, 47, 82, 32, 12, 5, 121, 27]
] # Magic numbers while creating the appropriate instances

S = range(len(deltasi))
VP = range(len(deltasi[0]))
VQ = range(len(deltaij[0]))
K = range(len(dK))

model = ConcreteModel()

model.gip = Var(VP, domain = Binary)
model.gjq = Var(VQ, domain = Binary)
model.xsi = Var(S, VP, domain = NonNegativeReals)
model.yij = Var(VP, VQ, domain = NonNegativeReals)
model.zjk = Var(VQ, K, domain = NonNegativeReals)
model.tiip = Var(VP, VP, domain = NonNegativeReals)

model.constraints = ConstraintList()

model.constraints.add(sum(model.gip[i] for i in VP) == rP)
model.constraints.add(sum(model.gjq[j] for j in VQ) == rQ)

for i in VP:
  model.constraints.add(sum(model.xsi[s, i] for s in S) <= sum(dK[k]*model.gip[i] for k in K))
  model.constraints.add(sum(model.yij[i, j] for j in VQ) <= sum(dK[k]*model.gip[i] for k in K))
  model.constraints.add(sum(model.tiip[i, ip] for ip in VP if i != ip) <= sum(dK[k]*model.gip[i] for k in K))
  model.constraints.add(sum(model.xsi[s, i] for s in S) + sum(model.tiip[ip, i] for ip in VP if i != ip) == sum(model.yij[i, j] for j in VQ) + sum(model.tiip[i, ip] for ip in VP if i != ip))
  model.constraints.add(sum(model.tiip[ip, i] for ip in VP if i != ip) == l*sum(model.yij[i, j] for j in VQ))

for ip in VP:
  model.constraints.add(sum(model.tiip[i, ip] for i in VP if i != ip) <= sum(dK[k]*model.gip[ip] for k in K))

for j in VQ:
  model.constraints.add(sum(model.yij[i, j] for i in VP) <= sum(dK[k]*model.gjq[j] for k in K))
  model.constraints.add(sum(model.zjk[j, k] for k in K) <= sum(dK[k]*model.gjq[j] for k in K))
  model.constraints.add(sum(model.yij[i, j] for i in VP) == sum(model.zjk[j, k] for k in K))

for k in K:
  model.constraints.add(sum(model.zjk[j, k] for j in VQ) == dK[k])

model.objective = Objective(
  expr = sum(sum(CS*deltasi[s][i]*model.xsi[s, i] for i in VP) for s in S) + sum(sum(CP*deltaij[i][j]*model.yij[i, j] for j in VQ) for i in VP) + sum(sum(CP*deltaiip[i][ip]*model.tiip[i, ip] for ip in VP if i != ip) for i in VP) + sum(sum(CK*deltajk[j][k]*model.zjk[j, k] for k in K) for j in VQ) + sum(ettaiP[i]*model.gip[i] for i in VP) + sum(ettajQ[j]*model.gjq[j] for j in VQ) + sum(gammaiP[i] * (sum(model.yij[i, j] for j in VQ) + sum(model.tiip[i, ip] for ip in VP if i != ip)) for i in VP) + sum(gammajQ[j] * sum(model.zjk[j, k] for k in K) for j in VQ),
  sense = minimize
)

results = SolverFactory('cplex').solve(model)
results.write()

#if results.solver.status:
#  model.pprint()

#model.constraints.display()
