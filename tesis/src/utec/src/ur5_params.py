import numpy as np

#Longitudes (en metros) del DH
l1 = 0.089159
l2 = 0.42500
l3 = 0.39225
l4 = 0.10915
l5 = 0.09465
l6 = 0.0823
l = [l1, l2, l3, l4, l5, l6]

#Limites articulares
q_lim = np.around(np.radians(np.array([[-360.0, 360.0],[-360.0, 360.0],[-180.0, 180.0],[-360.0, 360.0],[-360.0, 360.0],[-360.0, 360.0]])),4)
dq_lim = np.around(np.radians(np.array([[-180.0, 180.0],[-180.0, 180.0],[-180.0, 180.0],[-180.0, 180.0],[-180.0, 180.0],[-180.0, 180.0]])),4)

#Workspace
#R1 = 0.85 #Radio horizontal de la esfera
R1 = 0.875
#R2 = 0.8995 #Radio de la esfera
R2 = 0.975
ri = 0.149 #Radio del cilindro
r_ws = [R1, R2, ri]
