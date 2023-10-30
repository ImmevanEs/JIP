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

    
V_arr = np.linspace(0.7, 2, 6)
P_list = []
for i in range(len(V_arr)): 
    V_val = V_arr[i]
    print('v='+ str(V_val))
    TSR_arr = np.array([0.1, 0.3, 0.5, 0.7, 0.85, 0.9])
    F_list = []
    for i in range(len(TSR_arr)):
        TSR = TSR_arr[i]
        print('TSR = ' + str(TSR))
        #defining all the symbols/parameters
        #theta1, theta2, theta3, theta4, theta5 = me.dynamicsymbols('theta1, theta2, theta3, theta4, theta5') 
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
        #V = 2       #inflow velocity [m/s]    
        #TSR = 0.85      #tip speed ratio

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
        C1.orient_axis(N, N.z, -(sm.pi/2)+phi)
        C2.orient_axis(N, N.z, -(sm.pi/2)+(phi+(2*sm.pi/5)))
        C3.orient_axis(N, N.z, -(sm.pi/2)+(phi+ 2*(2*sm.pi/5)))
        C4.orient_axis(N, N.z, -(sm.pi/2)+(phi+ 3*(2*sm.pi/5)))
        C5.orient_axis(N, N.z, -(sm.pi/2)+(phi+ 4*(2*sm.pi/5)))
        D1.orient_axis(C1, C1.z, -delta1)
        D2.orient_axis(C2, C2.z, -delta2)
        D3.orient_axis(C3, C3.z, -delta3)
        D4.orient_axis(C4, C4.z, -delta4)
        D5.orient_axis(C5, C5.z, -delta5)

        #defining points
        #the point A1..A5 and C1..C5 represent the point of pressure of the airfoil. 
        #for simplicity, it is assumed that all the forces are applied at these point of pressures
        No =   sm.symbols('No', cls=me.Point)
        C1o, C2o, C3o, C4o, C5o = sm.symbols('C1o, C2o, C3o, C4o, C5o', cls=me.Point)
        D1o, D2o, D3o, D4o, D5o = sm.symbols('D1o, D2o, D3o, D4o, D5o', cls=me.Point)

        #defining the configuration 
        C1o.set_pos(No, r2*sm.cos(phi)*N.x + r2*sm.sin(phi)*N.y)
        D1o.set_pos(C1o, 0)
        C2o.set_pos(No, r2*sm.cos(phi+(2*sm.pi/5))*N.x + r2*sm.sin(phi+(2*sm.pi/5))*N.y)
        D2o.set_pos(C2o, 0)
        C3o.set_pos(No, r2*sm.cos(phi+ 2*(2*sm.pi/5))*N.x + r2*sm.sin(phi+ 2*(2*sm.pi/5))*N.y)
        D3o.set_pos(C3o, 0)
        C4o.set_pos(No, r2*sm.cos(phi+ 3*(2*sm.pi/5))*N.x + r2*sm.sin(phi+ 3*(2*sm.pi/5))*N.y)
        D4o.set_pos(C4o, 0)
        C5o.set_pos(No, r2*sm.cos(phi+ 4*(2*sm.pi/5))*N.x + r2*sm.sin(phi+ 4*(2*sm.pi/5))*N.y)
        D5o.set_pos(C5o, 0)

        #defining and setting the velocities 
        No.set_vel(N, 0)
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
        fh6 = (C1o.pos_from(No).dot(N.z))
        fh7 = (C2o.pos_from(No).dot(N.z))
        fh8 = (C3o.pos_from(No).dot(N.z))
        fh9 = (C4o.pos_from(No).dot(N.z))
        fh10 = (C5o.pos_from(No).dot(N.z))
        #constraining the points C1..C5 to be at a distance of r2 from the centre No
        fh16 = (C1o.pos_from(No).magnitude() - r2)
        fh17 = (C2o.pos_from(No).magnitude() - r2)
        fh18 = (C3o.pos_from(No).magnitude() - r2)
        fh19 = (C4o.pos_from(No).magnitude() - r2)
        fh20 = (C5o.pos_from(No).magnitude() - r2)
        #realise that the points are constrained to be equally spaced on the circumference in the definition of the configuration
        #creating one vector containing all constraints 
        fh = sm.Matrix([fh6, fh7, fh8, fh9, fh10, fh16, fh17, fh18, fh19, fh20])

        #computing relative velocities for different blades(reference frames A1..A5 and C1..C5)
        #computing the unit vector in the direction of the relative velocity
        #the relative velocities are a function of phi
        U2 = phi.diff(t)*r2
        #arm 1
        #blade 2
        v21 = V_val*N.x + U2*C1.x
        u_v21 = v21/v21.magnitude()
        #arm2
        #blade2
        v22 = V_val*N.x + U2*C2.x
        u_v22 = v22/v22.magnitude()
        #arm 3
        #blade 2
        v23 = V_val*N.x + U2*C3.x
        u_v23 = v23/v23.magnitude()
        #arm 4
        #blade 2
        v24 = V_val*N.x + U2*C4.x
        u_v24 = v24/v24.magnitude()
        #arm 5
        #blade 2
        v25 = V_val*N.x + U2*C5.x
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

        # =============================================================================
        # xp = np.linspace(0, 180, 36)
        # plt.plot(AoA, CoL, '.', xp, CL_fit(xp), '-')
        # plt.xlabel('AoA')
        # plt.ylabel('CL')
        # plt.suptitle('Polynomial fit of CL')
        # plt.show()
        # 
        # xp = np.linspace(0, 180, 36)
        # plt.plot(AoA, CoD, '.', xp, CD_fit(xp), '-')
        # plt.xlabel('AoA')
        # plt.ylabel('CD')
        # plt.suptitle('Polynomial fit of CD')
        # plt.show()
        # =============================================================================

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
        alpha21 = (sm.acos(v21.dot(D1.x)/(v21.magnitude()))).simplify()
        alpha22 = (sm.acos(v22.dot(D2.x)/(v22.magnitude()))).simplify()
        alpha23 = (sm.acos(v23.dot(D3.x)/(v23.magnitude()))).simplify()
        alpha24 = (sm.acos(v24.dot(D4.x)/(v24.magnitude()))).simplify()
        alpha25 = (sm.acos(v25.dot(D5.x)/(v25.magnitude()))).simplify()
        #alpha_mat = sm.Matrix([alpha11, alpha12, alpha13, alpha14, alpha15, alpha21, alpha22, alpha23, alpha24, alpha25])
        #print(alpha_mat)
        
        #Computing the lift and drag forces for all the blades
        #arm 1
        #blade 2
        Fl21 =  (1/2)*rho*sm.sqrt(v21.magnitude())*C*CL(alpha21*(180/sm.pi), CL_fit)*(u_v21.cross(N.z))
        Fd21 =  (1/2)*rho*sm.sqrt(v21.magnitude())*C*CD(alpha21*(180/sm.pi), CD_fit)*u_v21
        #arm 2 
        #blade 2
        Fl22 =  (1/2)*rho*sm.sqrt(v22.magnitude())*C*CL(alpha22*(180/sm.pi), CL_fit)*(u_v22.cross(N.z))
        Fd22 =  (1/2)*rho*sm.sqrt(v22.magnitude())*C*CD(alpha22*(180/sm.pi), CD_fit)*u_v22
        # #arm 3 
        #blade 2
        Fl23 =  (1/2)*rho*sm.sqrt(v23.magnitude())*C*CL(alpha23*(180/sm.pi), CL_fit)*(u_v23.cross(N.z))
        Fd23 =  (1/2)*rho*sm.sqrt(v23.magnitude())*C*CD(alpha23*(180/sm.pi), CD_fit)*u_v23
        #arm 24 
        #blade 2
        Fl24 =  (1/2)*rho*sm.sqrt(v24.magnitude())*C*CL(alpha24*(180/sm.pi), CL_fit)*(u_v24.cross(N.z))
        Fd24 =  (1/2)*rho*sm.sqrt(v24.magnitude())*C*CD(alpha24*(180/sm.pi), CD_fit)*u_v24
        #arm 5 
        #blade 2
        Fl25 =  (1/2)*rho*sm.sqrt(v25.magnitude())*C*CL(alpha25*(180/sm.pi), CL_fit)*(u_v25.cross(N.z))
        Fd25 =  (1/2)*rho*sm.sqrt(v25.magnitude())*C*CD(alpha25*(180/sm.pi), CD_fit)*u_v25



        #determining the resultant force per blade
        #arm 1
        F21 = Fl21+Fd21
        #arm 2
        F22 = Fl22+Fd22
        #arm 3
        F23 = Fl23+Fd23
        #arm 4
        F24 = Fl24+Fd24
        #arm 5
        F25 = Fl25+Fd25

        #projecting the resultant forces on the axis parallel to the rotational velocity vector
        #arm1 
        F_21 = F21.dot(-C1.x)
        #arm2 
        F_22 = F22.dot(-C2.x)
        #arm3 
        F_23 = F23.dot(-C3.x)
        #arm4 
        F_24 = F24.dot(-C4.x)
        #arm5 
        F_25 = F25.dot(-C5.x)

        #/////////////////////////////////////////////////////////////////////////////////////////////
        #determining the optimal pitch based on the simulation of one single arm 
        #this is determined by simulating only one arm with two blades
        #/////////////////////////////////////////////////////////////////////////////////////////////

        #determining the accumulative torque 
        F_torque = (F_21)*r2

        #lambdifying the torque for specific phi and over range of pitch
        pitch = np.linspace(0,2*np.pi,1000)
        F_torque = sm.lambdify((phi,phi.diff(t),delta1), F_torque)
        #alp = sm.lambdify((phi,phi.diff(t),theta1,delta1), alpha11)
        phis = np.arange(0,360,1)
        pitch_opt = []
        for i in range(len(phis)):
            F = F_torque(phis[i]*(np.pi/180),TSR*V_val/r2, pitch)
            peak, _ = signal.find_peaks(F)
            peak_max = peak[np.argmax(F[peak])]
            #x_peak_max = pitch[peak_max]
            pitch_opt.append(pitch[peak_max])

        #/////////////////////////////////////////////////////////////////////////////////////////////
        #/////////////////////////////////////////////////////////////////////////////////////////////

        #determining the accumulative torque 
        F_torque = (F_21 + F_22 +F_23 + F_24 + F_25)*r2
        f = sm.lambdify((phi, phi.diff(t), delta1, delta2, delta3, delta4, delta5), F_torque)

        phis = np.arange(0,2*np.pi,2*np.pi/360)
        def pitch_angle(phis,arms):
            phis = phis*(180/np.pi)
            phis = round(phis + arms*72)
            while phis >= 360:
                phis = round(phis - 360) 
            return pitch_opt[phis]   
        
        #computing the optimal pitch for each blade
        delta1 = []
        delta2 = []
        delta3 = []
        delta4 = []
        delta5 = []
        for i in range(len(phis)):
            delta1.append(pitch_angle(phis[i], 0))    
            delta2.append(pitch_angle(phis[i], 1))
            delta3.append(pitch_angle(phis[i], 2))
            delta4.append(pitch_angle(phis[i], 3))
            delta5.append(pitch_angle(phis[i], 4))
            
        #inserting all the values and getting a value for F
        F = f(phis, TSR*V_val/r2, delta1, delta2, delta3, delta4, delta5)

        #for a V and a specific TSR calculate the mean
        F_mean = statistics.mean(F)
        #add mean to the list
        F_list.append(F_mean) 
        
    
    #The resultant power is the list of torque of all TSR of one V        
    F_arr = np.array(F_list)
    #the power of the all the TSR for one V
    P = F_arr*TSR_arr*V_val/r2
        
    #the plot of P vs TSR for every V
    #plt.plot(TSR_arr, P)
    #plt.xlabel('TSR [-]')
    #plt.ylabel('Power [W]')
    #plt.suptitle('Power curve versus TSR')
    #plt.show()
    
    #adding a row for each velocity containing all the torques for specific TSR   
    P_list.append(P)
P_arr = np.array(P_list)
#plt.plot(V_arr, P_arr)
#plt.xlabel('Velocity [m/s]')
#plt.ylabel('Power [W]')
#plt.suptitle('Power curve versus velocity')
#plt.show()

#plot for power vs TSR
plt.plot(TSR_arr, P_arr[0], label='Velocity =' +str(V_arr[0])+ '[m/s]')
plt.plot(TSR_arr, P_arr[1], label='Velocity =' +str(V_arr[1])+ '[m/s]')
plt.plot(TSR_arr, P_arr[2], label='Velocity =' +str(V_arr[2])+ '[m/s]')
plt.plot(TSR_arr, P_arr[3], label='Velocity =' +str(V_arr[3])+ '[m/s]')
plt.plot(TSR_arr, P_arr[4], label='Velocity =' +str(V_arr[4])+ '[m/s]')
plt.plot(TSR_arr, P_arr[5], label='Velocity =' +str(V_arr[5])+ '[m/s]')
plt.xlabel('TSR [-]')
plt.ylabel('Power [W]')
plt.suptitle('Power curve versus TSR for different velocities for one blade per arm')
plt.legend( bbox_to_anchor=(1.45, 0.5), loc='center right')
plt.show()

#plot for power vs velocity 
plt.plot(V_arr, P_arr.T[0], label='TSR =' + str(TSR_arr[0]) + '[m/s]')
plt.plot(V_arr, P_arr.T[1], label='TSR =' + str(TSR_arr[1])+ '[m/s]')
plt.plot(V_arr, P_arr.T[2], label='TSR =' + str(TSR_arr[2])+ '[m/s]')
plt.plot(V_arr, P_arr.T[3], label='TSR =' + str(TSR_arr[3])+ '[m/s]')
plt.plot(V_arr, P_arr.T[4], label='TSR =' + str(TSR_arr[4])+ '[m/s]')
plt.plot(V_arr, P_arr.T[5], label='TSR =' + str(TSR_arr[5])+ '[m/s]')
plt.xlabel('Velocity [m/s]')
plt.ylabel('Power [W]')
plt.suptitle('Power curve versus velocity for different TSR for one blade per arm')
plt.legend( bbox_to_anchor=(1.37, 0.5), loc='center right')
plt.show()

#plotting the Cp vs TSR
C_P_list = []
for i in range(len(V_arr)):
    V = V_arr[i]
    C_P = (F_arr*TSR_arr*V/r2)/(0.5*rho*V**3*(2*r2)*b)
    C_P_list.append(C_P)

C_P_arr = np.array(C_P_list)
plt.plot(TSR_arr, C_P_arr[0], label = 'Velocity =' +str(V_arr[5]))
plt.plot(TSR_arr, C_P_arr[1], label = 'Velocity =' +str(V_arr[4]))
plt.plot(TSR_arr, C_P_arr[2], label = 'Velocity =' +str(V_arr[3]))
plt.plot(TSR_arr, C_P_arr[3], label = 'Velocity =' +str(V_arr[2]))
plt.plot(TSR_arr, C_P_arr[4], label = 'Velocity =' +str(V_arr[1]))
plt.plot(TSR_arr, C_P_arr[5], label = 'Velocity =' +str(V_arr[0]))
plt.xlabel('TSR [-]')
plt.ylabel('Power coefficient [-]')
plt.suptitle('Power coefficient versus TSR for different velocities')
plt.legend( bbox_to_anchor=(1.35, 0.5), loc='center right')
plt.show()



















    




