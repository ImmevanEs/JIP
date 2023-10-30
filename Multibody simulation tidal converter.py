# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 13:16:09 2023

Calculation of performance of tidal converter
Using multibody simulation
NEMO by ArianeGroup

@author: immev
"""


import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from sympy import *
import matplotlib.pyplot as plt
import pandas as pd
import csv
import subprocess
from subprocess import Popen, PIPE
from scipy import optimize
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp, odeint
import scipy.integrate as integrate
import scipy.special as special
from scipy.integrate import quad, dblquad
from sympy.vector import vector_integrate, CoordSys3D
from sympy import collect, trigsimp, cancel
import gurobipy as gp
from gurobipy import Model, GRB, quicksum
from scipy import optimize, signal
import statistics

#defining all the symbols/parameters
theta1, theta2, theta3, theta4, theta5 = me.dynamicsymbols('theta1, theta2, theta3, theta4, theta5') 
delta1, delta2, delta3, delta4, delta5 = me.dynamicsymbols('delta1, delta2, delta3, delta4, delta5')
#psi1, psi2, psi3, psi4, psi5 = me.dynamicsymbols('psi1, psi2, psi3, psi4, psi5')
phi = me.dynamicsymbols('phi')
alpha = me.dynamicsymbols('alpha')
Fl1, Fl2, Fl3, Fl4, Fl5, Fl6, Fl7, Fl8, Fl9, Fl10 = me.dynamicsymbols('Fl1, Fl2, Fl3, Fl4, Fl5, Fl6, Fl7, Fl8, Fl9, Fl10')  
Fd1, Fd2, Fd3, Fd4, Fd5, Fd6, Fd7, Fd8, Fd9, Fd10 = me.dynamicsymbols('Fd1, Fd2, Fd3, Fd4, Fd5, Fd6, Fd7, Fd8, Fd9, Fd10') 
r1, r2 = sm.symbols('r1, r2')
d, C, w = sm.symbols('d, C, w')
U1, U2, V = sm.symbols('U1, U2, V')
rho, mu = sm.symbols('rho, mu')
t = me.dynamicsymbols._t


#Input parameters
#flow
rho = 1029     #density [kg/m^3]
V = 2       #inflow velocity [m/s]    
TSR = 0.85      #tip speed ratio

#blade
C = 0.5        #chorld length [m] 
b = 1.6        #draft [m]
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
A2 = me.ReferenceFrame('A2')
A3 = me.ReferenceFrame('A3')
A4 = me.ReferenceFrame('A4')
A5 = me.ReferenceFrame('A5')
B1 = me.ReferenceFrame('B1')
B2 = me.ReferenceFrame('B2')
B3 = me.ReferenceFrame('B3')
B4 = me.ReferenceFrame('B4')
B5 = me.ReferenceFrame('B5')
C1 = me.ReferenceFrame('C1')
C2 = me.ReferenceFrame('C2')
C3 = me.ReferenceFrame('C3')
C4 = me.ReferenceFrame('C4')
C5 = me.ReferenceFrame('C5')
D1 = me.ReferenceFrame('D1')
D2 = me.ReferenceFrame('D2')
D3 = me.ReferenceFrame('D3')
D4 = me.ReferenceFrame('D4')
D5 = me.ReferenceFrame('D5')

#rotating the reference with the right angles 
A1.orient_axis(N, N.z, -(sm.pi/2)+phi)
A2.orient_axis(N, N.z, -(sm.pi/2)+(phi+(2*sm.pi/5))) #or phi2 ??????????
A3.orient_axis(N, N.z, -(sm.pi/2)+(phi+ 2*(2*sm.pi/5)))
A4.orient_axis(N, N.z, -(sm.pi/2)+(phi+ 3*(2*sm.pi/5)))
A5.orient_axis(N, N.z, -(sm.pi/2)+(phi+ 4*(2*sm.pi/5)))
C1.orient_axis(N, N.z, -(sm.pi/2)+phi)
C2.orient_axis(N, N.z, -(sm.pi/2)+(phi+(2*sm.pi/5)))
C3.orient_axis(N, N.z, -(sm.pi/2)+(phi+ 2*(2*sm.pi/5)))
C4.orient_axis(N, N.z, -(sm.pi/2)+(phi+ 3*(2*sm.pi/5)))
C5.orient_axis(N, N.z, -(sm.pi/2)+(phi+ 4*(2*sm.pi/5)))
B1.orient_axis(A1, A1.z, -theta1)
B2.orient_axis(A2, A2.z, -theta2)
B3.orient_axis(A3, A3.z, -theta3)
B4.orient_axis(A4, A4.z, -theta4)
B5.orient_axis(A5, A5.z, -theta5)
D1.orient_axis(C1, C1.z, -delta1)
D2.orient_axis(C2, C2.z, -delta2)
D3.orient_axis(C3, C3.z, -delta3)
D4.orient_axis(C4, C4.z, -delta4)
D5.orient_axis(C5, C5.z, -delta5)

#defining points
#the point A1..A5 and C1..C5 represent the point of pressure of the airfoil. 
#for simplicity, it is assumed that all the forces are applied at these point of pressures
No =   sm.symbols('No', cls=me.Point)
A1o, A2o, A3o, A4o, A5o = sm.symbols('A1o, A2o, A3o, A4o, A5o', cls=me.Point)
B1o, B2o, B3o, B4o, B5o = sm.symbols('B1o, B2o, B3o, B4o, B5o', cls=me.Point)
C1o, C2o, C3o, C4o, C5o = sm.symbols('C1o, C2o, C3o, C4o, C5o', cls=me.Point)
D1o, D2o, D3o, D4o, D5o = sm.symbols('D1o, D2o, D3o, D4o, D5o', cls=me.Point)

#defining the configuration 
A1o.set_pos(No, r1*sm.cos(phi)*N.x + r1*sm.sin(phi)*N.y)
B1o.set_pos(A1o, 0)
C1o.set_pos(A1o, r2*sm.cos(phi)*N.x + r2*sm.sin(phi)*N.y)
D1o.set_pos(C1o, 0)
A2o.set_pos(No, r1*sm.cos(phi+(2*sm.pi/5))*N.x + r1*sm.sin(phi+(2*sm.pi/5))*N.y)
B2o.set_pos(A2o, 0)
C2o.set_pos(No, r2*sm.cos(phi+(2*sm.pi/5))*N.x + r2*sm.sin(phi+(2*sm.pi/5))*N.y)
D2o.set_pos(C2o, 0)
A3o.set_pos(No, r1*sm.cos(phi+ 2*(2*sm.pi/5))*N.x + r1*sm.sin(phi+ 2*(2*sm.pi/5))*N.y)
B3o.set_pos(A3o, 0)
C3o.set_pos(No, r2*sm.cos(phi+ 2*(2*sm.pi/5))*N.x + r2*sm.sin(phi+ 2*(2*sm.pi/5))*N.y)
D3o.set_pos(C3o, 0)
A4o.set_pos(No, r1*sm.cos(phi+ 3*(2*sm.pi/5))*N.x + r1*sm.sin(phi+ 3*(2*sm.pi/5))*N.y)
B4o.set_pos(A4o, 0)
C4o.set_pos(No, r2*sm.cos(phi+ 3*(2*sm.pi/5))*N.x + r2*sm.sin(phi+ 3*(2*sm.pi/5))*N.y)
D4o.set_pos(C4o, 0)
A5o.set_pos(No, r1*sm.cos(phi+ 4*(2*sm.pi/5))*N.x + r1*sm.sin(phi+ 4*(2*sm.pi/5))*N.y)
B5o.set_pos(A5o, 0)
C5o.set_pos(No, r2*sm.cos(phi+ 4*(2*sm.pi/5))*N.x + r2*sm.sin(phi+ 4*(2*sm.pi/5))*N.y)
D5o.set_pos(C5o, 0)

#defining and setting the velocities 
No.set_vel(N, 0)
A1o.set_vel(A1, 0)
A1o.v1pt_theory(No, N, A1)
A2o.set_vel(A2, 0)
A2o.v1pt_theory(No, N, A2)
A3o.set_vel(A3, 0)
A3o.v1pt_theory(No, N, A3)
A4o.set_vel(A4, 0)
A4o.v1pt_theory(No, N, A4)
A5o.set_vel(A5, 0)
A5o.v1pt_theory(No, N, A5)
C1o.set_vel(C1, 0)
C1o.v1pt_theory(No, N, C1)
C2o.set_vel(C2, 0)
C2o.v1pt_theory(No, N, C2)
C3o.set_vel(C3, 0)
C3o.v1pt_theory(No, N, C3)
C4o.set_vel(C4, 0)
C4o.v1pt_theory(No, N, C4)
C5o.set_vel(C5, 0)
C5o.v1pt_theory(No, N, C5)

#constraining the points/configuration 
#constraining the points A1..A5 and C1..C5 to remain in the N.x,N.y-plane
fh1 = (A1o.pos_from(No).dot(N.z))
fh2 = (A2o.pos_from(No).dot(N.z))
fh3 = (A3o.pos_from(No).dot(N.z))
fh4 = (A4o.pos_from(No).dot(N.z))
fh5 = (A5o.pos_from(No).dot(N.z))
fh6 = (C1o.pos_from(No).dot(N.z))
fh7 = (C2o.pos_from(No).dot(N.z))
fh8 = (C3o.pos_from(No).dot(N.z))
fh9 = (C4o.pos_from(No).dot(N.z))
fh10 = (C5o.pos_from(No).dot(N.z))
#constraining the points A1..A5 to be at a distance of r1 from the centre No
fh11 = (A1o.pos_from(No).magnitude() - r1)
fh12 = (A2o.pos_from(No).magnitude() - r1)
fh13 = (A3o.pos_from(No).magnitude() - r1)
fh14 = (A4o.pos_from(No).magnitude() - r1)
fh15 = (A5o.pos_from(No).magnitude() - r1)
#constraining the points C1..C5 to be at a distance of r2 from the centre No
fh16 = (C1o.pos_from(No).magnitude() - r2)
fh17 = (C2o.pos_from(No).magnitude() - r2)
fh18 = (C3o.pos_from(No).magnitude() - r2)
fh19 = (C4o.pos_from(No).magnitude() - r2)
fh20 = (C5o.pos_from(No).magnitude() - r2)
#realise that the points are constrained to be equally spaced on the circumference in the definition of the configuration
#creating one vector containing all constraints 
fh = sm.Matrix([fh1, fh2, fh3, fh4, fh5, fh6, fh7, fh8, fh9, fh10, fh11, fh12, fh13, fh14, fh15, fh16, fh17, fh18, fh19, fh20])

#computing relative velocities for different blades(reference frames A1..A5 and C1..C5)
#computing the unit vector in the direction of the relative velocity
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
#arm2
#blade1
v12 = V*N.x + U1*A2.x
u_v12 = v12/v12.magnitude()
#blade2
v22 = V*N.x + U2*C2.x
u_v22 = v22/v22.magnitude()
#arm 3
#blade 1
v13 = V*N.x + U1*A3.x
u_v13 = v13/v13.magnitude()
#blade 2
v23 = V*N.x + U2*C3.x
u_v23 = v23/v23.magnitude()
#arm 4
#blade 1
v14 = V*N.x + U1*A4.x
u_v14 = v14/v14.magnitude()
#blade 2
v24 = V*N.x + U2*C4.x
u_v24 = v24/v24.magnitude()
#arm 5
#blade 1
v15 = V*N.x + U1*A5.x
u_v15 = v15/v15.magnitude()
#blade 2
v25 = V*N.x + U2*C5.x
u_v25 = v25/v25.magnitude()

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
plt.xlabel('AoA')
plt.ylabel('CL')
plt.suptitle('Polynomial fit of CL')
plt.show()

xp = np.linspace(0, 180, 36)
plt.plot(AoA, CoD, '.', xp, CD_fit(xp), '-')
plt.xlabel('AoA')
plt.ylabel('CD')
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
alpha12 = (sm.acos(v12.dot(B2.x)/(v12.magnitude()))).simplify()
alpha13 = (sm.acos(v13.dot(B3.x)/(v13.magnitude()))).simplify()
alpha14 = (sm.acos(v14.dot(B4.x)/(v14.magnitude()))).simplify()
alpha15 = (sm.acos(v15.dot(B5.x)/(v15.magnitude()))).simplify()
alpha21 = (sm.acos(v21.dot(D1.x)/(v21.magnitude()))).simplify()
alpha22 = (sm.acos(v22.dot(D2.x)/(v22.magnitude()))).simplify()
alpha23 = (sm.acos(v23.dot(D3.x)/(v23.magnitude()))).simplify()
alpha24 = (sm.acos(v24.dot(D4.x)/(v24.magnitude()))).simplify()
alpha25 = (sm.acos(v25.dot(D5.x)/(v25.magnitude()))).simplify()

#Computing the lift and drag forces for all the blades
#arm 1
#blade 1
Fl11 =  (1/2)*rho*sm.sqrt(v11.magnitude())*C*CL(alpha11*(180/sm.pi), CL_fit)*(u_v11.cross(N.z))
Fd11 =  (1/2)*rho*sm.sqrt(v11.magnitude())*C*CD(alpha11*(180/sm.pi), CD_fit)*u_v11
# #blade 2
Fl21 =  (1/2)*rho*sm.sqrt(v21.magnitude())*C*CL(alpha21*(180/sm.pi), CL_fit)*(u_v21.cross(N.z))
Fd21 =  (1/2)*rho*sm.sqrt(v21.magnitude())*C*CD(alpha21*(180/sm.pi), CD_fit)*u_v21
#arm 2 
#blade1
Fl12 =  (1/2)*rho*sm.sqrt(v12.magnitude())*C*CL(alpha12*(180/sm.pi), CL_fit)*(u_v12.cross(N.z))
Fd12 =  (1/2)*rho*sm.sqrt(v12.magnitude())*C*CD(alpha12*(180/sm.pi), CD_fit)*u_v12
# #blade 2
Fl22 =  (1/2)*rho*sm.sqrt(v22.magnitude())*C*CL(alpha22*(180/sm.pi), CL_fit)*(u_v22.cross(N.z))
Fd22 =  (1/2)*rho*sm.sqrt(v22.magnitude())*C*CD(alpha22*(180/sm.pi), CD_fit)*u_v22
# #arm 3 
# #blade1
Fl13 =  (1/2)*rho*sm.sqrt(v13.magnitude())*C*CL(alpha13*(180/sm.pi), CL_fit)*(u_v13.cross(N.z))
Fd13 =  (1/2)*rho*sm.sqrt(v13.magnitude())*C*CD(alpha13*(180/sm.pi), CD_fit)*u_v13
# #blade 2
Fl23 =  (1/2)*rho*sm.sqrt(v23.magnitude())*C*CL(alpha23*(180/sm.pi), CL_fit)*(u_v23.cross(N.z))
Fd23 =  (1/2)*rho*sm.sqrt(v23.magnitude())*C*CD(alpha23*(180/sm.pi), CD_fit)*u_v23
#arm 24 
#blade1
Fl14 =  (1/2)*rho*sm.sqrt(v14.magnitude())*C*CL(alpha14*(180/sm.pi), CL_fit)*(u_v14.cross(N.z))
Fd14 =  (1/2)*rho*sm.sqrt(v14.magnitude())*C*CD(alpha14*(180/sm.pi), CD_fit)*u_v14
#blade 2
Fl24 =  (1/2)*rho*sm.sqrt(v24.magnitude())*C*CL(alpha24*(180/sm.pi), CL_fit)*(u_v24.cross(N.z))
Fd24 =  (1/2)*rho*sm.sqrt(v24.magnitude())*C*CD(alpha24*(180/sm.pi), CD_fit)*u_v24
#arm 5 
#blade1
Fl15 =  (1/2)*rho*sm.sqrt(v15.magnitude())*C*CL(alpha15*(180/sm.pi), CL_fit)*(u_v15.cross(N.z))
Fd15 =  (1/2)*rho*sm.sqrt(v15.magnitude())*C*CD(alpha15*(180/sm.pi), CD_fit)*u_v15
#blade 2
Fl25 =  (1/2)*rho*sm.sqrt(v25.magnitude())*C*CL(alpha25*(180/sm.pi), CL_fit)*(u_v25.cross(N.z))
Fd25 =  (1/2)*rho*sm.sqrt(v25.magnitude())*C*CD(alpha25*(180/sm.pi), CD_fit)*u_v25

#determining the resultant force per blade
#arm 1
F11 = Fl11+Fd11
F21 = Fl21+Fd21
#arm 2
F12 = Fl12+Fd12
F22 = Fl22+Fd22
#arm 3
F13 = Fl13+Fd13
F23 = Fl23+Fd23
#arm 4
F14 = Fl14+Fd14
F24 = Fl24+Fd24
#arm 5
F15 = Fl15+Fd15
F25 = Fl25+Fd25

#projecting the resultant forces on the axis parallel to the rotational velocity vector
#arm1 
F_11 = F11.dot(-A1.x)
F_21 = F21.dot(-C1.x)
#arm2 
F_12 = F12.dot(-A2.x)
F_22 = F22.dot(-C2.x)
#arm3 
F_13 = F13.dot(-A3.x)
F_23 = F23.dot(-C3.x)
#arm4 
F_14 = F14.dot(-A4.x)
F_24 = F24.dot(-C4.x)
#arm5 
F_15 = F15.dot(-A5.x)
F_25 = F25.dot(-C5.x)

#/////////////////////////////////////////////////////////////////////////////////////////////
#determining the optimal pitch based on the simulation of one single arm 
#this is determined by simulating only one arm with two blades
#/////////////////////////////////////////////////////////////////////////////////////////////

#determining the accumulative torque for one arm
F_torque = (F_11)*r1 + (F_21)*r2

#lambdifying the torque for specific phi and over range of pitch
#finding the maximum torque at each position of phi and finding the corresponding value for pitch
#the pitch is saved and returned
pitch = np.linspace(0,2*np.pi,1000)
F_torque = sm.lambdify((phi,phi.diff(t),theta1,delta1), F_torque)
phis = np.arange(0,360,1)
pitch_opt = []
for i in range(len(phis)):
    F = F_torque(phis[i]*(np.pi/180),TSR*V/r2, pitch, pitch)
    peak, _ = signal.find_peaks(F)
    peak_max = peak[np.argmax(F[peak])]
    pitch_opt.append(pitch[peak_max])

#/////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////

#determining the accumulative torque for all the blades
F_torque = (F_11 +  F_12 + F_13 + F_14 + F_15)*r1 + (F_21 + F_22 +F_23 + F_24 + F_25)*r2
f = sm.lambdify((phi, phi.diff(t), theta1, theta2, theta3, theta4, theta5, delta1, delta2, delta3, delta4, delta5), F_torque)

#defining a function of the pitch angle in terms of phi and the arm
phis = np.arange(0,2*np.pi,2*np.pi/360)
def pitch_angle(phis,arms):
    phis = phis*(180/np.pi)
    phis = round(phis + arms*72)
    while phis >= 360:
        phis = round(phis - 360) 
    return pitch_opt[phis]   

#computing the optimal pitch for each blade
theta1 = []
theta2 = []
theta3 = []
theta4 = []
theta5 = []
delta1 = []
delta2 = []
delta3 = []
delta4 = []
delta5 = []
for i in range(len(phis)):
    theta1.append(pitch_angle(phis[i], 0))    
    theta2.append(pitch_angle(phis[i], 1))
    theta3.append(pitch_angle(phis[i], 2))
    theta4.append(pitch_angle(phis[i], 3))
    theta5.append(pitch_angle(phis[i], 4))
    delta1.append(pitch_angle(phis[i], 0))    
    delta2.append(pitch_angle(phis[i], 1))
    delta3.append(pitch_angle(phis[i], 2))
    delta4.append(pitch_angle(phis[i], 3))
    delta5.append(pitch_angle(phis[i], 4))

#inserting all the values and getting a value for F
F = f(phis, TSR*V/r2, theta1, theta2, theta3, theta4, theta5, delta1, delta2, delta3, delta4, delta5)
#determining the mean of the force 
F_mean = statistics.mean(F)

ptch = np.array(pitch_opt)
#account for the fact that the numerical solution does not take into account head and tail so difference of 180
#the graph for optimal pitch is in the one arm simulation file 
for i in range(len(phis)):
     if phis[i] < 80*(np.pi/180):
         ptch[i] = ptch[i] + np.pi
     else:
         ptch[i] = ptch[i]

#creating lists for theta1 .. theta5 and delta1...delta5 for the plot
theta1, delta1 = ptch, ptch
theta2 = []
delta2 = []
for i in range(len(theta1)):
    x = theta1[i] + 72*(np.pi/180)
    while x >= 2*np.pi:
        x = x - 2*np.pi
    theta2.append(x)
    delta2.append(x)
theta2 = np.array(theta2)
delta2 = np.array(delta2)
theta3 = []
delta3 = []
for i in range(len(theta2)):
    x = theta2[i] + 72*(np.pi/180)
    while x >= 2*np.pi:
        x = x - 2*np.pi
    theta3.append(x)
    delta3.append(x)
theta3 = np.array(theta3)
delta3 = np.array(delta3)
theta4 = []
delta4 = []
for i in range(len(theta3)):
    x = theta3[i] + 72*(np.pi/180)
    while x >= 2*np.pi:
        x = x - 2*np.pi
    theta4.append(x)
    delta4.append(x)
theta4 = np.array(theta4)
delta4 = np.array(delta4)
theta5 = []
delta5 = []
for i in range(len(theta4)):
    x = theta4[i] + 72*(np.pi/180)
    while x >= 2*np.pi:
        x = x - 2*np.pi
    theta5.append(x)
    delta5.append(x)
theta5 = np.array(theta5)
delta5 = np.array(delta5)

#plot for optimal pitch vs phi
plt.plot(phis*180/np.pi, ptch*180/np.pi)
plt.xlabel('Phis [deg]')
plt.ylabel('Optimal pitch [deg]')
plt.suptitle('Optimal pitch for a single arm for all position in the rotor')
plt.show()

#plot for optimal pitch vs phi (specified for each individual blade)
plt.plot(phis*180/np.pi, theta1*180/np.pi, label='pitch blades arm 1')
plt.plot(phis*180/np.pi, theta2*180/np.pi, label='pitch blades arm 2')
plt.plot(phis*180/np.pi, theta3*180/np.pi, label='pitch blades arm 3')
plt.plot(phis*180/np.pi, theta4*180/np.pi, label='pitch blades arm 4')
plt.plot(phis*180/np.pi, theta5*180/np.pi, label='pitch blades arm 5')
plt.xlabel('Phis (degrees)')
plt.ylabel('Optimal pitch (degrees)')
plt.suptitle('Optimal pitch for all blades for all position in the rotor')
plt.legend( bbox_to_anchor=(1.42, 0.5), loc='center right')
plt.show()

#plot for torque vs phi
plt.plot(phis*180/np.pi,F)
plt.xlabel('Phi (degrees)')
plt.ylabel('Torque (Nm)')
plt.suptitle('Torque against phi for optimized pitched blades with TSR is 0.8')
plt.show()





