# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 08:06:33 2023

@author: immev
"""
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
import matplotlib.pyplot as plt
import csv

#simulation of just one arm of the tidal converter

#defining all the symbols/parameters
theta1 = me.dynamicsymbols('theta1') 
delta1 = me.dynamicsymbols('delta1')
#psi1, psi2, psi3, psi4, psi5 = me.dynamicsymbols('psi1, psi2, psi3, psi4, psi5')
phi = me.dynamicsymbols('phi')
alpha = me.dynamicsymbols('alpha')
r1, r2 = sm.symbols('r1, r2')
d, C, w = sm.symbols('d, C, w')
U1, U2, V = sm.symbols('U1, U2, V')
rho, mu = sm.symbols('rho, mu')
t = me.dynamicsymbols._t

#Input parameters
#flow
rho = 1029     #density [kg/m^3]
V = 2       #inflow velocity [m/s]    
TSR = 0.85      #tip speed ratio [-]

#blade
C = 0.5        #chorld length [m] 
b = 1.6        #span [m]
#Ap = 0.8      #planform [m^2]
Z = 5          #number of blades

#rotor configuration
r1 = 6.1       #inner radius [m]
r2 = 8.45      #outer radius [m]


#defining reference frames 
#every blade has 2 additional referenceframes. 
#(1) that is always perpendicular to the orbit of rotation, thus can be used conveniently for normal and tangential forces
#(2) that is body fixed and thus can be used conveniently for lift and drag forces
N = me.ReferenceFrame('N')
A1 = me.ReferenceFrame('A1')
B1 = me.ReferenceFrame('B1')
C1 = me.ReferenceFrame('C1')
D1 = me.ReferenceFrame('D1')

#rotating the reference with the right angles 
A1.orient_axis(N, N.z, -(sm.pi/2)+phi)
C1.orient_axis(N, N.z, -(sm.pi/2)+phi)
B1.orient_axis(A1, A1.z, -theta1)
D1.orient_axis(C1, C1.z, -delta1)

#defining points
#the point A1..A5 and C1..C5 represent the point of pressure of the airfoil. 
#for simplicity, it is assumed that all the forces are applied at these point of pressures
No = sm.symbols('No', cls=me.Point)
A1o = sm.symbols('A1o', cls=me.Point)
B1o = sm.symbols('B1o', cls=me.Point)
C1o = sm.symbols('C1o', cls=me.Point)
D1o = sm.symbols('D1o', cls=me.Point)

#defining the configuration 
A1o.set_pos(No, r1*sm.cos(phi)*N.x + r1*sm.sin(phi)*N.y)
B1o.set_pos(A1o, 0)
C1o.set_pos(No, r2*sm.cos(phi)*N.x + r2*sm.sin(phi)*N.y)
D1o.set_pos(C1o, 0)

#defining and setting the velocities 
No.set_vel(N, 0)
A1o.set_vel(A1, 0)
A1o.v1pt_theory(No, N, A1)
C1o.set_vel(C1, 0)
C1o.v1pt_theory(No, N, C1)

#constraining the points/configuration 
#constraining the points A1..A5 and C1..C5 to remain in the N.x,N.y-plane
fh1 = (A1o.pos_from(No).dot(N.z))
fh6 = (C1o.pos_from(No).dot(N.z))
#constraining the points A1..A5 to be at a distance of r1 from the centre No
fh11 = (A1o.pos_from(No).magnitude() - r1)
#constraining the points C1..C5 to be at a distance of r2 from the centre No
fh16 = (C1o.pos_from(No).magnitude() - r2)

#computing relative velocities for different blades(reference frames A1..A5 and C1..C5)
#the relative velocities are a function of phi
U1 = phi.diff(t)*r1
U2 = phi.diff(t)*r2
#arm 1
#blade 1
v11 = V*N.x + U1*A1.x
u_v11 = v11/v11.magnitude()
#blade 2
v21 = V*N.x + U2*C1.x
u_v21 = v21/v21.magnitude()

#Computing the reynolds number for the system
#the density of sea water * the flow speed * chord length / dynamic viscosity
Re = rho*V*C/mu
#the reynolds number influences the cd and cl curves 

#Computing an equation for Cl and Cd as a function of the angle of attack 
#using data from [Aerodynamic Characteristics of Seven Symmetrical Airfoil Sections Through 180-Degree Angle of Attack for Use in Aerodynamic Analysis of Vertical Axis Wind Turbines]
#for the NACA 0012 airfoil 
data_NACA0012 = open('data cd cl NACA0012.csv', 'r')
data = csv.DictReader(data_NACA0012)
AoA = []
CoL = []
CoD = []
for col in data:
    AoA.append(col['AoA'])
    CoL.append(col['Cl'])
    CoD.append(col['Cd'])
for i, item in enumerate(AoA):
    AoA[i] = eval(item)
for i, item in enumerate(CoL):
    CoL[i] = eval(item)
for i, item in enumerate(CoD):
    CoD[i] = eval(item)    
angleofattack = np.array(AoA)
CL = np.array(CoL)
CD = np.array(CoD)
CL_fit = np.polyfit(AoA, CL, 10)
CL_fit = np.poly1d(CL_fit)
CD_fit = np.polyfit(AoA, CD, 5)
CD_fit = np.poly1d(CD_fit)


#plotting the curve of CL and CD
xp = np.linspace(0, 180, 36)
plt.plot(AoA, CoL, '.', xp, CL_fit(xp), '-')
plt.xlabel('Angle of attack [degrees]')
plt.ylabel('Coefficient of lift [-]')
plt.suptitle('Polynomial fit of CL')
plt.show()

xp = np.linspace(0, 180, 36)
plt.plot(AoA, CoD, '.', xp, CD_fit(xp), '-')
plt.xlabel('Angle of attack [degrees]')
plt.ylabel('Coefficient of drag [-]')
plt.suptitle('Polynomial fit of CD')
plt.show()

#creating a function for CL and CD, so that a value for CD and CL can be determined based on the angle of attack
def CL(alpha, CL_fit):
    CL =  0
    for a in range(len(CL_fit)):
        CL = CL + CL_fit[len(CL_fit)-a]*alpha**(len(CL_fit)-a)
    return CL

def CD(alpha, CD_fit):
    CD =  0
    for a in range(len(CD_fit)):
        CD = CD + CD_fit[len(CD_fit)-a]*alpha**(len(CD_fit)-a)
    return CD

#writing the angle fo attack for all the blades as a function of phi, the first derivatiev of phi and the second derivative of phi
alpha11 = (sm.acos(v11.dot(B1.x)/(v11.magnitude()))).simplify()
alpha21 = (sm.acos(v21.dot(B1.x)/(v21.magnitude()))).simplify()

#arm 1
#blade 1
Fl11 =  (1/2)*rho*((v11.magnitude())**2)*C*CL(alpha11*(180/sm.pi), CL_fit)*(u_v11.cross(N.z))
Fd11 =  (1/2)*rho*((v11.magnitude())**2)*C*CD(alpha11*(180/sm.pi), CD_fit)*u_v11
# #blade 2
Fl21 =  (1/2)*rho*((v21.magnitude())**2)*C*CL(alpha21*(180/sm.pi), CL_fit)*(u_v21.cross(N.z))
Fd21 =  (1/2)*rho*((v21.magnitude())**2)*C*CD(alpha21*(180/sm.pi), CD_fit)*u_v21

#computing the contribution to the torque of the lift and drag force
F_lift = (Fl11.dot(-A1.x))*r1 + (Fl21.dot(-C1.x))*r2
F_drag = (Fd11.dot(-A1.x))*r1 + (Fd21.dot(-C1.x))*r2 

f_l = sm.lambdify((phi, phi.diff(t), theta1, delta1), F_lift)
f_d = sm.lambdify((phi, phi.diff(t), theta1, delta1), F_drag)

#determining the resultant force per blade
#arm 1
F11 = Fl11+Fd11
F21 = Fl21+Fd21

#projecting the resultant forces on the axis parallel to the rotational velocity vector
#arm1 
F_11 = F11.dot(-A1.x)
F_21 = F21.dot(-C1.x)

#determining the accumulative torque 
F_torque = (F_11)*r1 + (F_21)*r2

#lambdifying the torque for specific phi and over range of pitch
#finding the maximum torque at each position of phi and finding the corresponding value for pitch
#the pitch is saved and returned
pitch = np.linspace(0,2*np.pi,360)
F_torque = sm.lambdify((phi,phi.diff(t),theta1,delta1), F_torque)
phis = np.arange(0,360,1)
pitch_opt = []
for i in range(len(phis)):
    F = F_torque(phis[i]*(np.pi/180),TSR*V/r2, pitch, pitch)
    peak, _ = signal.find_peaks(F)
    peak_max = peak[np.argmax(F[peak])]
    pitch_opt.append(pitch[peak_max])

ptch = np.array(pitch_opt)
for i in range(len(phis)):
     if phis[i] < 80:
         ptch[i] = ptch[i] + np.pi
     else:
         ptch[i] = ptch[i]

#plot for optimal pitch vs phi
plt.plot(phis, ptch*180/np.pi)
plt.xlabel('Phis (degrees)')
plt.ylabel('Optimal pitch (degrees)')
plt.show()

#plotting the of the lift and drag force to the torque
F_L = f_l(phis, TSR*V/r2, ptch, ptch)
F_D = f_d(phis, TSR*V/r2, ptch, ptch)

plt.plot(phis, F, label= 'Total torque force')
plt.plot(phis, F_L, label='Torque due to lift force')
plt.plot(phis, F_D, label='Torque due to drag force')
plt.xlabel('Phis (degrees)')
plt.ylabel('Torque [Nm]')
plt.legend( bbox_to_anchor=(1.3, 0.5), loc='center right')
plt.suptitle('Contribution of lift and drag force to torque')
plt.show()



