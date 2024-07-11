import math
import numpy as np

from copy import copy
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

q_lim = np.around(np.radians(np.array([[-360.0, 360.0],[-360.0, 360.0],[-180.0, 180.0],[-360.0, 360.0],[-360.0, 360.0],[-360.0, 360.0]])),4)

def dh(d, theta, a, alpha):
    """
    Calcular la matriz de transformacion homogenea asociada con los parametros
    de Denavit-Hartenberg.
    Los valores d, theta, a, alpha son escalares.

    """
    
    sth = np.sin(theta)
    cth = np.cos(theta)
    sa  = np.sin(alpha)
    ca  = np.cos(alpha)
    T = np.array([[cth, -ca*sth,  sa*sth, a*cth],
                  [sth,  ca*cth, -sa*cth, a*sth],
                  [0.0,      sa,      ca,     d],
                  [0.0,     0.0,     0.0,   1.0]])
                  
    return T
    
def fkine(q, l):
	"""
	Calcular la cinematica directa del robot UR dados sus valores articulares. 
	q es un vector numpy de la forma [q1, q2, q3, q4, q5, q6]

	"""
	
	# Matrices DH (completar), emplear la funcion dh con los parametros DH para cada articulacion
	T1 = dh(l[0], q[0], 0, np.pi/2)
	T2 = dh(0, q[1], -l[1], 0)
	T3 = dh(0, q[2], -l[2], 0)
	T4 = dh(l[3], q[3], 0, np.pi/2)
	T5 = dh(l[4], q[4], 0, -np.pi/2)
	T6 = dh(l[5], q[5], 0, 0)
	# Efector final con respecto a la base
	T = T1.dot(T2).dot(T3).dot(T4).dot(T5).dot(T6)

	return T
	
def jacobian_position(q, l, delta=0.0001):
	"""
	Jacobiano analitico para la posicion. Retorna una matriz de 3x6 y toma como
	entrada el vector de configuracion articular q=[q1, q2, q3, q4, q5, q6]

	"""
	
	# Alocacion de memoria
	J = np.zeros((3,6))
	# Transformacion homogenea inicial (usando q)
	T = fkine(q,l)
	# Iteracion para la derivada de cada columna
	for i in range(6):
		# Copiar la configuracion articular inicial
		dq = copy(q)
		# Incrementar la articulacion i-esima usando un delta
		dq[i] = dq[i] + delta
		# Transformacion homogenea luego del incremento (q+dq)
		Td = fkine(dq,l)
		# Aproximacion del Jacobiano de posicion usando diferencias	finitas
		Ji = (Td[0:3,3] - T[0:3,3])/delta
		J[:,i] = Ji

	return J
	
def ikine(xdes, q0, l):
	"""
	Calcular la cinematica inversa de UR numericamente a partir de la
	configuracion articular inicial de q0.
	"""
	
	#Parametros
	epsilon = 0.0001
	max_iter = 1000
	delta = 0.00001
	
	# Copiar la configuracion articular inicial
	q = copy(q0)
	
	# Main loop
	for i in range(max_iter):
	
		#Calcular jacobiano analitico
		J = jacobian_position(q,l,delta)
		#Hallar la posicion del efector final
		T06_q = fkine(q,l)
		x = T06_q[0:3,3]
		#Error de posicion
		e = xdes-x
		#Condicion de termino
		if (np.linalg.norm(e) < epsilon):
			break
		#Actualizar valores de q
		q = q + np.dot(np.linalg.pinv(J), e)
	
	return q
	
def jacobian_pose(q, l, delta=0.0001):
	"""
	Jacobiano analitico para la posicion y orientacion (usando un
	cuaternion). Retorna una matriz de 7x6 y toma como entrada el vector de
	configuracion articular q=[q1, q2, q3, q4, q5, q6]

	"""
	
	# Alocacion de memoria
	J = np.zeros((7,6))
	# Transformacion homogenea inicial (usando q)
	T = fkine(q,l)
	# Iteracion para la derivada de cada columna
	for i in range(6):
		# Copiar la configuracion articular inicial (usar este dq para cada incremento en una articulacion)
		dq = copy(q)
		# Incrementar la articulacion i-esima usando un delta
		dq[i] = dq[i] + delta
		# Transformacion homogenea luego del incremento (q+dq)
		Td = fkine(dq,l)
		# Aproximacion del Jacobiano de posicion usando diferencias finitas
		Ji_p = (Td[0:3,3] - T[0:3,3])/delta
		Ji_o = (R.from_matrix(Td[0:3,0:3]).as_quat() - R.from_matrix(T[0:3,0:3]).as_quat())/delta
		J[0:3,i] = Ji_p
		J[3:,i] = Ji_o

	return J
	
def Jacob_inv_singular(J, k=0.1):
	"""
	Calcula la pseudo-inversa del Jacobiano dependiendo de si es singular
	"""
	
	#Si J no es singular (rango igual a 6), se usa la pseudo-inversa de Moore Penrose
	if np.linalg.matrix_rank(J) == 6:
		Jinv = np.linalg.pinv(J)
	#Si J es singular (rango menor a 6), se usa la pseudo-inversa amortiguada
	elif np.linalg.matrix_rank(J) < 6:
		Jinv = np.dot(J.T,np.linalg.inv(np.dot(J,J.T)+k**2*np.eye(6)))
		
	return Jinv

def calc_eo_quat(Q, Qd):
	"""
	Calcula el error de orientacion entre dos cuaterniones
	Entradas:
	Q -- Cuaternion de orientacion actual
	Qd -- Cuaternion de orientacion deseada
	Salida:
	e_o -- Error de orientacion
	"""
	
	#Calculo de Qe = Qd * Q(-1)
	Q = np.roll(Q,1)
	Qd = np.roll(Qd,1)
	Qact = Quaternion(Q)
	Qdes = Quaternion(Qd)
	Qerr = Qdes.__mul__(Qact.inverse)
	Qe = Qerr.elements
	
	#Error de orientacion
	eo_q = Qe - np.array([1,0,0,0])
	eo_q = np.roll(eo_q,-1)

	return eo_q

def calc_eo_quat_2(Q, Qd):
	"""
	Calcula el error de orientacion entre dos cuaterniones
	Entradas:
	Q -- Cuaternion de orientacion actual
	Qd -- Cuaternion de orientacion deseada
	Salida:
	e_o -- Error de orientacion
	"""
	
	Qe_e = Qd[3]*Q[0:3] - Q[3]*Qd[0:3] - np.cross(Qd[0:3],Q[0:3])
	Qe_w = Qd[3]*Q[3] + np.dot(Qd[0:3],Q[0:3])
	Qe = np.array([Qe_e[0],Qe_e[1],Qe_e[2],Qe_w])
	
	return Qe

def TF2xyzquat(T):
    """
    Convert a homogeneous transformation matrix into the a vector containing the
    pose of the robot.

    Input:
      T -- A homogeneous transformation
    Output:
      X -- A pose vector in the format [x y z ew ex ey ez], donde la first part
           is Cartesian coordinates and the last part is a quaternion
    """
    quat = R.from_matrix(T[0:3,0:3]).as_quat()
    res = [T[0,3], T[1,3], T[2,3], quat[0], quat[1], quat[2], quat[3]]
    return np.array(res)
	
def limit_joint_pos(q, q_lim):
	"""
	Delimita los valores articulares a los limites articulares del UR5
	"""
	
	#Verifica si cada valor articular esta dentro de sus limites
	for i in range(6):
		if q[i] < q_lim[i,0]:
			q[i] = q_lim[i,0]
		elif q[i] > q_lim[i,1]:
			q[i] = q_lim[i,1]
		else:
			q[i] = q[i]

	return q
	
def limit_joint_vel(dq, dq_lim):
	"""
	Delimita los valores articulares a los limites articulares del UR5
	"""
	
	#Verifica si cada valor articular esta dentro de sus limites
	for i in range(6):
		if dq[i] < dq_lim[i,0]:
			dq[i] = dq_lim[i,0]
		elif dq[i] > dq_lim[i,1]:
			dq[i] = dq_lim[i,1]
		else:
			dq[i] = dq[i]

	return dq
	
def check_ur_ws(x_des, r_ws):
	"""
	Verifica si la posicion deseada no sale del espacio de trabajo del UR.
	"""
	
	#Coordenadas de la posicion deseada
	x = x_des[0]
	y = x_des[1]
	z = x_des[2]
	
	
	#Radios del workspace
	R1 = r_ws[0]
	R2 = r_ws[1]
	ri = r_ws[2]
		
	check = False
	
	if ((x/R1)**2+(y/R1)**2+(z/R2)**2 <= 1 and math.sqrt(x**2+y**2) >= ri) and (z >= 0):
		check = True
	else:
		check = False
	
	return check
	
def ikine_pose(pose_des, q0, l):
	"""
	Calcular la cinematica inversa de UR numericamente a partir de la
	configuracion articular inicial de q0.
	"""
	
	#Parametros
	epsilon = 1e-3
	max_iter = 10000
	delta = 0.0001
	
	# Copiar la configuracion articular inicial
	q = copy(q0)
	
	# Main loop
	for i in range(max_iter):
	
		#Calcular jacobiano analitico
		J = jacobian_pose(q,l,delta)
		#Hallar la posicion y orientacion del efector final
		T06_q = fkine(q,l)
		x = T06_q[0:3,3]
		Q = R.from_matrix(T06_q[0:3,0:3]).as_quat()
		#Error de posicion y orientacion
		e_pos = x-pose_des[0:3]
		e_o = calc_eo_quat(Q,pose_des[3:])
		e = -10*np.concatenate((e_pos,-e_o))
		print(np.linalg.norm(e))
		#Condicion de termino
		if (np.linalg.norm(e) < epsilon):
			break
		#Actualizar valores de q
		q = q + 0.01*np.dot(Jacob_inv_singular(J), e)
		q = limit_joint_pos(q,q_lim)
	
	return q
