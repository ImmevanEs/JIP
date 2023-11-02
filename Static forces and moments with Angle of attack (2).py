# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:05:21 2023

@author: Vincent Erd
"""

import math
from colorama import Fore
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
import matplotlib.pyplot as plt
import pandas as pd
import csv
from tabulate import tabulate
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import scipy.integrate as integrate
import scipy.special as special
from scipy.integrate import quad
import operator

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
CD_fit = np.polyfit(AoA, CD, 10)
CD_fit = np.poly1d(CD_fit)

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

#computing the forces 
#realise that CL_fit and CD_fit are a function of AoA alpha
#writing the functions of CL_fit and CD_fit explicitely as a polynomial of alpha
        
#variables Blade & Water
Rho     = 1029                          #[Kg/m^3]
A       = 10                             #[m^2]


#Definition rotor arm
alpha = 48                                              #Angle of Y connection point wrt horizontal
i = 3.7                                                 #Length of diagonal part of rotorarm
c = i * math.cos(math.radians(alpha))                   #Horizontal length of diagonal part rotorarm
d = 5.77                                                #Horizonal length of rotorarm
L = c + d                                               #Length total Arm [m]
a =  9                                                  #Length of outer blade to center [m]
b =  6.08                                               #Length of inner blade to center [m]
e = 2.75                                                #Hight of Y connection to top of arm
f = 7.25                                                #Length of outer blade to Y connection
g = 4.33                                                #Length of inner blade to Y connection
h = 1.75                                                #Length of Y connection point to center

F_lift_T1 = {}
F_drag_T1 = {}
F_lift_P1 = {}
F_drag_P1 = {}
F_lift_T2 = {}
F_drag_T2 = {}
F_lift_P2 = {}
F_drag_P2 = {}
F_Tangent1 = {}
F_Tangent2 = {}
M_Center = {}
M_Center_MAX = {}
M_Center_MIN = {}
Mp_Y_point = {}
Mp_Y_point_MAX = {}
Mp_Y_point_MIN = {}
Mt_Y_point = {}
Mt_Y_point_MAX = {}
Mt_Y_point_MIN = {}
F_Tangent_1 = {}
F_Tangent_2 = {}
M_Tangent_Tot = {}
C_l1 = {}
C_d1 = {}
C_l2 = {}
C_d2 = {}
F_P_1= {}
F_P_2 = {}
F_P_Tot = {}
F_P_MaxTot = {}
F_P_MinTot = {}
F_1_Max = {}
F_2_Max = {}
M_Tangent_Tot_MAX = {}
M_Tangent_Tot_MIN = {}

Angles = pd.read_excel(r"Rotation-pitch angel.xlsx")

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

x = range (360)
for i in x:
    C_l1[i]= CL(Angles.iat[i,6],CL_fit)
    C_d1[i]= CD(Angles.iat[i,6],CD_fit)
    C_l2[i]= CL(Angles.iat[i,11],CL_fit)
    C_d2[i]= CD(Angles.iat[i,11],CD_fit)
    
x = range(360)                                                                                  #Tangential forces due to drag (negative counter works rotation)
for i in x:
    if 0 <= Angles.iat[i,6] <=90:
        F_drag_T1[i] = -1/2*Rho*A*(Angles.iat[i,7]**2)*math.cos(math.radians(Angles.iat[i,1]))*abs(C_d1[i])
        
    if 90<= Angles.iat[i,6] <=180:
        F_drag_T1[i] = 1/2*Rho*A*(Angles.iat[i,7]**2)*math.cos(math.radians(Angles.iat[i,1]))*abs(C_d1[i])    
    
x = range(360)                                                                                  #Tangential forces due to drag (negative counter works rotation)
for i in x:
    if 0 <= Angles.iat[i,11] <=90:
        F_drag_T2[i] = -1*(1/2*Rho*A*(Angles.iat[i,12]**2)*math.cos(math.radians(Angles.iat[i,1])))*abs(C_d2[i])
        
    if 90<= Angles.iat[i,11] <=180:
        F_drag_T2[i] = 1/2*Rho*A*(Angles.iat[i,12]**2)*math.cos(math.radians(Angles.iat[i,1]))*abs(C_d2[i]) 
        
x = range(360)                                                                                  #Perpendicular to ration forces due to drag (negative is compression)
for i in x:
    if Angles.iat[i,1] >= 0 and Angles.iat[i,6] <= 90:
        F_drag_P1[i] = -1/2*Rho*A*(Angles.iat[i,7]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_d1[i]) #Compression
    
    if Angles.iat[i,1] <= 0 and Angles.iat[i,6] >= 90:
        F_drag_P1[i] = -1/2*Rho*A*(Angles.iat[i,7]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_d1[i]) #Compression
    
    if Angles.iat[i,1] >= 0 and Angles.iat[i,6] >= 90:
        F_drag_P1[i] = 1/2*Rho*A*(Angles.iat[i,7]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_d1[i])  #Pull
        
    if Angles.iat[i,1] <= 0 and Angles.iat[i,6] <= 90:
        F_drag_P1[i] = 1/2*Rho*A*(Angles.iat[i,7]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_d1[i])  #Pull

x = range(360)                                                                                  #Perpendicular to ration forces due to drag (negative is compression)
for i in x:
    if Angles.iat[i,1] >= 0 and Angles.iat[i,11] <= 90:
        F_drag_P2[i] = -1/2*Rho*A*(Angles.iat[i,12]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_d2[i]) #Compression
    
    if Angles.iat[i,1] <= 0 and Angles.iat[i,11] >= 90:
        F_drag_P2[i] = -1/2*Rho*A*(Angles.iat[i,12]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_d2[i]) #Compression
    
    if Angles.iat[i,1] >= 0 and Angles.iat[i,11] >= 90:
        F_drag_P2[i] = 1/2*Rho*A*(Angles.iat[i,12]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_d2[i])   #Pull
     
    if Angles.iat[i,1] <= 0 and Angles.iat[i,11] <= 90:
         F_drag_P2[i] = 1/2*Rho*A*(Angles.iat[i,12]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_d2[i])  #Pull   
     
x = range(360)
for i in x:
    if 0 <= Angles.iat[i,0] <= 90:
        if 0 <= Angles.iat[i,1] <= Angles.iat[i,4]:
            F_lift_T1[i] = 1/2*Rho*A*(Angles.iat[i,7]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_l1[i])    
            F_lift_P1[i] = -1/2*Rho*A*(Angles.iat[i,7]**2)*math.cos(math.radians(Angles.iat[i,1]))*abs(C_l1[i])   
        if 0 <= Angles.iat[i,1] >= Angles.iat[i,4]:
            F_lift_T1[i] = -1/2*Rho*A*(Angles.iat[i,7]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_l1[i])
            F_lift_P1[i] = 1/2*Rho*A*(Angles.iat[i,7]**2)*math.cos(math.radians(Angles.iat[i,1]))*abs(C_l1[i])   
        if 0 >= Angles.iat[i,1]:
            F_lift_T1[i] = -1/2*Rho*A*(Angles.iat[i,7]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_l1[i])
            F_lift_P1[i] = -1/2*Rho*A*(Angles.iat[i,7]**2)*math.cos(math.radians(Angles.iat[i,1]))*abs(C_l1[i])   

    if 90 <= Angles.iat[i,0] <= 180:
        if Angles.iat[i,4] <= 90:
            if 0 <= Angles.iat[i,1] <= Angles.iat[i,4]:
                F_lift_T1[i] = 1/2*Rho*A*(Angles.iat[i,7]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_l1[i])    
                F_lift_P1[i] = -1/2*Rho*A*(Angles.iat[i,7]**2)*math.cos(math.radians(Angles.iat[i,1]))*abs(C_l1[i])   
            if 0 <= Angles.iat[i,1] >= Angles.iat[i,4]:
                F_lift_T1[i] = -1/2*Rho*A*(Angles.iat[i,7]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_l1[i])    
                F_lift_P1[i] = 1/2*Rho*A*(Angles.iat[i,7]**2)*math.cos(math.radians(Angles.iat[i,1]))*abs(C_l1[i])
            if 0 >= Angles.iat[i,1]:
                F_lift_T1[i] = -1/2*Rho*A*(Angles.iat[i,7]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_l1[i])    
                F_lift_P1[i] = -1/2*Rho*A*(Angles.iat[i,7]**2)*math.cos(math.radians(Angles.iat[i,1]))*abs(C_l1[i])
        if Angles.iat[i,4] >= 90:
            if 0 <= Angles.iat[i,1]:
                F_lift_T1[i] = 1/2*Rho*A*(Angles.iat[i,7]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_l1[i])    
                F_lift_P1[i] = -1/2*Rho*A*(Angles.iat[i,7]**2)*math.cos(math.radians(Angles.iat[i,1]))*abs(C_l1[i])   
            if 0 >= Angles.iat[i,1] and Angles.iat[i,4] <= 180 + Angles.iat[i,1]:
                F_lift_T1[i] = -1/2*Rho*A*(Angles.iat[i,7]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_l1[i])    
                F_lift_P1[i] = -1/2*Rho*A*(Angles.iat[i,7]**2)*math.cos(math.radians(Angles.iat[i,1]))*abs(C_l1[i])
            if 0 >= Angles.iat[i,1] and Angles.iat[i,4] >= 180 + Angles.iat[i,1]:
                F_lift_T1[i] = 1/2*Rho*A*(Angles.iat[i,7]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_l1[i])    
                F_lift_P1[i] = 1/2*Rho*A*(Angles.iat[i,7]**2)*math.cos(math.radians(Angles.iat[i,1]))*abs(C_l1[i])
        
    if 180 <= Angles.iat[i,0] <= 270:
        if Angles.iat[i,4] + Angles.iat[i,1] >= 180:
            F_lift_T1[i] = 1/2*Rho*A*(Angles.iat[i,7]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_l1[i])
            F_lift_P1[i] = -1/2*Rho*A*(Angles.iat[i,7]**2)*math.cos(math.radians(Angles.iat[i,1]))*abs(C_l1[i])
        if Angles.iat[i,4] + Angles.iat[i,1] <= 180:
            if Angles.iat[i,1] >= 0:
                F_lift_T1[i] = -1/2*Rho*A*(Angles.iat[i,7]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_l1[i])
                F_lift_P1[i] = 1/2*Rho*A*(Angles.iat[i,7]**2)*math.cos(math.radians(Angles.iat[i,1]))*abs(C_l1[i])
            if Angles.iat[i,1] <= 0:
                F_lift_T1[i] = 1/2*Rho*A*(Angles.iat[i,7]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_l1[i])
                F_lift_P1[i] = 1/2*Rho*A*(Angles.iat[i,7]**2)*math.cos(math.radians(Angles.iat[i,1]))*abs(C_l1[i])
    
    if 270 <= Angles.iat[i,0] <= 360:
        if 0 >= Angles.iat[i,1]:
            if Angles.iat[i,4] >= abs(Angles.iat[i,1]):
                F_lift_T1[i] = 1/2*Rho*A*(Angles.iat[i,7]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_l1[i])
                F_lift_P1[i] = 1/2*Rho*A*(Angles.iat[i,7]**2)*math.cos(math.radians((Angles.iat[i,1])))*abs(C_l1[i])  
            if Angles.iat[i,4] <= abs(Angles.iat[i,1]):
                F_lift_T1[i] = -1/2*Rho*A*(Angles.iat[i,7]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_l1[i])
                F_lift_P1[i] = -1/2*Rho*A*(Angles.iat[i,7]**2)*math.cos(math.radians((Angles.iat[i,1])))*abs(C_l1[i])
        if 0 <= Angles.iat[i,1]:
            F_lift_T1[i] = -1/2*Rho*A*(Angles.iat[i,7]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_l1[i]) 
            F_lift_P1[i] = 1/2*Rho*A*(Angles.iat[i,7]**2)*math.cos(math.radians((Angles.iat[i,1])))*abs(C_l1[i])

x = range(360)
for i in x:
    if 0 <= Angles.iat[i,0] <= 90:
        if 0 <= Angles.iat[i,1] <= Angles.iat[i,9]:
            F_lift_T2[i] = 1/2*Rho*A*(Angles.iat[i,12]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_l2[i])    
            F_lift_P2[i] = -1/2*Rho*A*(Angles.iat[i,12]**2)*math.cos(math.radians(Angles.iat[i,1]))*abs(C_l2[i])   
        if 0 <= Angles.iat[i,1] >= Angles.iat[i,9]:
            F_lift_T2[i] = -1/2*Rho*A*(Angles.iat[i,12]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_l2[i])
            F_lift_P2[i] = 1/2*Rho*A*(Angles.iat[i,12]**2)*math.cos(math.radians(Angles.iat[i,1]))*abs(C_l2[i])   
        if 0 >= Angles.iat[i,1]:
            F_lift_T2[i] = -1/2*Rho*A*(Angles.iat[i,12]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_l2[i])
            F_lift_P2[i] = -1/2*Rho*A*(Angles.iat[i,12]**2)*math.cos(math.radians(Angles.iat[i,1]))*abs(C_l2[i])   

    if 90 <= Angles.iat[i,0] <= 180:
        if Angles.iat[i,9] <= 90:
            if 0 <= Angles.iat[i,1] <= Angles.iat[i,9]:
                F_lift_T2[i] = 1/2*Rho*A*(Angles.iat[i,12]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_l2[i])    
                F_lift_P2[i] = -1/2*Rho*A*(Angles.iat[i,12]**2)*math.cos(math.radians(Angles.iat[i,1]))*abs(C_l2[i])   
            if 0 <= Angles.iat[i,1] >= Angles.iat[i,9]:
                F_lift_T2[i] = -1/2*Rho*A*(Angles.iat[i,12]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_l2[i])    
                F_lift_P2[i] = 1/2*Rho*A*(Angles.iat[i,12]**2)*math.cos(math.radians(Angles.iat[i,1]))*abs(C_l2[i])
            if 0 >= Angles.iat[i,1]:
                F_lift_T2[i] = -1/2*Rho*A*(Angles.iat[i,12]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_l2[i])    
                F_lift_P2[i] = -1/2*Rho*A*(Angles.iat[i,12]**2)*math.cos(math.radians(Angles.iat[i,1]))*abs(C_l2[i])
        if Angles.iat[i,9] >= 90:
            if 0 <= Angles.iat[i,1]:
                F_lift_T2[i] = 1/2*Rho*A*(Angles.iat[i,12]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_l2[i])    
                F_lift_P2[i] = -1/2*Rho*A*(Angles.iat[i,12]**2)*math.cos(math.radians(Angles.iat[i,1]))*abs(C_l2[i])   
            if 0 >= Angles.iat[i,1] and Angles.iat[i,9] <= 180 + Angles.iat[i,1]:
                F_lift_T2[i] = -1/2*Rho*A*(Angles.iat[i,12]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_l2[i])    
                F_lift_P2[i] = -1/2*Rho*A*(Angles.iat[i,12]**2)*math.cos(math.radians(Angles.iat[i,1]))*abs(C_l2[i])
            if 0 >= Angles.iat[i,1] and Angles.iat[i,9] >= 180 + Angles.iat[i,1]:
                F_lift_T2[i] = 1/2*Rho*A*(Angles.iat[i,12]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_l2[i])    
                F_lift_P2[i] = 1/2*Rho*A*(Angles.iat[i,12]**2)*math.cos(math.radians(Angles.iat[i,1]))*abs(C_l2[i])
        
    if 180 <= Angles.iat[i,0] <= 270:
        if Angles.iat[i,9] + Angles.iat[i,1] >= 180:
            F_lift_T2[i] = 1/2*Rho*A*(Angles.iat[i,12]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_l2[i])
            F_lift_P2[i] = -1/2*Rho*A*(Angles.iat[i,12]**2)*math.cos(math.radians(Angles.iat[i,1]))*abs(C_l2[i])
        if Angles.iat[i,9] + Angles.iat[i,1] <= 180:
            if Angles.iat[i,1] >= 0:
                F_lift_T2[i] = -1/2*Rho*A*(Angles.iat[i,12]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_l2[i])
                F_lift_P2[i] = 1/2*Rho*A*(Angles.iat[i,12]**2)*math.cos(math.radians(Angles.iat[i,1]))*abs(C_l2[i])
            if Angles.iat[i,1] <= 0:
                F_lift_T2[i] = 1/2*Rho*A*(Angles.iat[i,12]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_l2[i])
                F_lift_P2[i] = 1/2*Rho*A*(Angles.iat[i,12]**2)*math.cos(math.radians(Angles.iat[i,1]))*abs(C_l2[i])
    
    if 270 <= Angles.iat[i,0] <= 360:
        if 0 >= Angles.iat[i,1]:
            if Angles.iat[i,9] >= abs(Angles.iat[i,1]):
                F_lift_T2[i] = 1/2*Rho*A*(Angles.iat[i,12]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_l2[i])
                F_lift_P2[i] = 1/2*Rho*A*(Angles.iat[i,12]**2)*math.cos(math.radians((Angles.iat[i,1])))*abs(C_l2[i])  
            if Angles.iat[i,9] <= abs(Angles.iat[i,1]):
                F_lift_T2[i] = -1/2*Rho*A*(Angles.iat[i,12]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_l2[i])
                F_lift_P2[i] = -1/2*Rho*A*(Angles.iat[i,12]**2)*math.cos(math.radians((Angles.iat[i,1])))*abs(C_l2[i])
        if 0 <= Angles.iat[i,1]:
            F_lift_T2[i] = -1/2*Rho*A*(Angles.iat[i,12]**2)*math.sin(math.radians(abs(Angles.iat[i,1])))*abs(C_l2[i]) 
            F_lift_P2[i] = 1/2*Rho*A*(Angles.iat[i,12]**2)*math.cos(math.radians((Angles.iat[i,1])))*abs(C_l2[i])

x = range(72)
for i in x:
    
    F_Tangent_1[i] = F_lift_T1[i]+F_drag_T1[i]+F_lift_T1[i+72]+F_drag_T1[i+72]+F_lift_T1[i+144]+F_drag_T1[i+144]+F_lift_T1[i+216]+F_drag_T1[i+216]+F_lift_T1[i+288]+F_drag_T1[i+288]
    F_Tangent_2[i] = F_lift_T2[i]+F_drag_T2[i]+F_lift_T2[i+72]+F_drag_T2[i+72]+F_lift_T2[i+144]+F_drag_T2[i+144]+F_lift_T2[i+216]+F_drag_T2[i+216]+F_lift_T2[i+288]+F_drag_T2[i+288]
    M_Tangent_Tot[i] = F_Tangent_1[i]*a + F_Tangent_2[i]*b
x = range(360)
for i in x:
    F_P_1[i]  = F_drag_P1[i]+F_lift_P1[i]
    F_P_2[i]  = F_drag_P2[i]+F_lift_P2[i]
    F_P_Tot[i] = F_P_1[i]+F_P_2[i]
    F_Tangent1[i] = F_lift_T1[i]+F_drag_T1[i]
    F_Tangent2[i] = F_lift_T2[i]+F_drag_T2[i]
    M_Center[i] = F_Tangent1[i]*a+F_Tangent2[i]*b
    Mp_Y_point[i] = (F_P_1[i]+F_P_2[i])*e
    Mt_Y_point[i] = (F_Tangent1[i]*f)+(F_Tangent2[i]*g)
    F_1_Max[i] = (F_P_1[i]**2+F_Tangent1[i]**2)**1/2
    F_2_Max[i] = (F_P_2[i]**2+F_Tangent2[i]**2)**1/2    
   
F_1_Max = (max(F_1_Max.items(), key=operator.itemgetter(1)))    
F_2_Max = (max(F_2_Max.items(), key=operator.itemgetter(1)))   
F_P_Max1 = (max(F_P_1.items(), key=operator.itemgetter(1)))
F_P_Max2 = (max(F_P_2.items(), key=operator.itemgetter(1)))
F_P_Min1 = (min(F_P_1.items(), key=operator.itemgetter(1)))
F_P_Min2 = (min(F_P_2.items(), key=operator.itemgetter(1)))
F_P_MaxTot = (max(F_P_Tot.items(), key=operator.itemgetter(1)))
F_P_MinTot = (min(F_P_Tot.items(), key=operator.itemgetter(1)))
M_Center_MAX = (max(M_Center.items(), key=operator.itemgetter(1)))
M_Center_MIN = (min(M_Center.items(), key=operator.itemgetter(1)))
Mp_Y_point_MAX = (max(Mp_Y_point.items(), key=operator.itemgetter(1)))
Mp_Y_point_MIN = (min(Mp_Y_point.items(), key=operator.itemgetter(1)))
Mt_Y_point_MAX = (max(Mt_Y_point.items(), key=operator.itemgetter(1)))
Mt_Y_point_MIN = (min(Mt_Y_point.items(), key=operator.itemgetter(1)))
M_Tangent_Tot_MAX = (max(M_Tangent_Tot.items(), key=operator.itemgetter(1)))
M_Tangent_Tot_MIN = (min(M_Tangent_Tot.items(), key=operator.itemgetter(1)))

print(Fore.GREEN + "Tension and compression forces on blade")
print(Fore.RESET + 'Maximum tension force on outer blade: [Deg, Newton]',F_P_Max1)
print('Maximum compression force on outer blade: [Deg, Newton]',F_P_Min1)
print('Maximum tension force on inner blade: [Deg, Newton]',F_P_Max2)
print('Maximum compression force on inner blade: [Deg, Newton]',F_P_Min2)
print(Fore.GREEN + "Maximal force on blade as combination of lift and drag")
print(Fore.RESET + 'Maximum force combination on outer blades: [Deg, Newton]',F_1_Max)
print('Maximum force combination on inner blade: [Deg, Newton]',F_2_Max)
print(Fore.GREEN + "tension and compression forces")
print(Fore.RESET + 'Maximum tension on rotor arm: [Deg, Newton',F_P_MaxTot)
print('Maximum compression on rotor arm: [Deg, Newton]',F_P_MinTot)
print(Fore.RED + 'Turning momentum at center')
print(Fore.RESET + 'Maximum turning momentum at center [Deg, Newtonmeter]', M_Center_MAX)
print('Maximum counter turning momentum at center [Deg, Newtonmeter]', M_Center_MIN)
print(Fore.RED + 'Turning momentum in Y point due to tangent forces')
print(Fore.RESET + 'Maximum turning momentum in rotational direction at Y connection point [Deg, Newtonmeter]',Mt_Y_point_MAX)
print('Maximum counter rotation turning momentum at Y connection point [Deg, Newtonmeter]',Mt_Y_point_MIN)
print(Fore.RED + 'Momentum in Y point due to tension and compression')
print(Fore.RESET + 'Maximum Tension moment in Y connection point [Deg, Newtonmeter]',Mp_Y_point_MAX)
print('Maximum Compression moment in Y connection point [Deg, Newtonmeter]',Mp_Y_point_MIN)

print(Fore.CYAN + 'Rotational momentum generated by all 5 blades')
print('Maximum rotational momentum generated by all five blades',M_Tangent_Tot_MAX)
print('Maximum counter rotational momentum generated by all five blades',M_Tangent_Tot_MIN)
x =  range(72)
for i in x:
    print(Fore.RESET +'Rotational momentum at',i,'degreess:',M_Tangent_Tot[i],'Nm')


plt.plot(list(F_drag_T1.keys()), list(F_drag_T1.values()), label = 'F_dragT1')
plt.plot(list(F_drag_T2.keys()), list(F_drag_T2.values()), label = 'F_dragT2')
plt.xlabel('Orientation of rotor arm')
plt.ylabel('Drag coefficient [-]')
plt.suptitle('Tangential drag forces ')
plt.legend( bbox_to_anchor=(1.3, 0.5), loc='center right')
plt.show()

plt.plot(list(F_lift_T1.keys()), list(F_lift_T1.values()), label = 'F_liftT1')
plt.plot(list(F_lift_T2.keys()), list(F_lift_T2.values()), label = 'F_liftT2')
plt.xlabel('Orientation of rotor arm')
plt.ylabel('Drag coefficient [-]')
plt.suptitle('Tangential lift forces ')
plt.legend( bbox_to_anchor=(1.3, 0.5), loc='center right')
plt.show()

plt.plot(list(F_drag_P1.keys()), list(F_drag_P1.values()), label = 'F_dragP1')
plt.plot(list(F_drag_P2.keys()), list(F_drag_P2.values()), label = 'F_dragP2')
plt.xlabel('Orientation of rotor arm')
plt.ylabel('Drag coefficient [-]')
plt.suptitle('Parallel drag forces ')
plt.legend( bbox_to_anchor=(1.3, 0.5), loc='center right')
plt.show()

plt.plot(list(F_P_Tot.keys()), list(F_P_Tot.values()))
plt.xlabel('Orientation of rotor arm')
plt.ylabel('Tenstion force [N]')
plt.suptitle('Tension and compression forces on rotor arm ')
plt.show()

plt.plot(list(M_Center.keys()), list(M_Center.values()))
plt.xlabel('Orientation of rotor arm')
plt.ylabel('Momentum [Nm]')
plt.suptitle('Rotational momentum of one arm')
plt.show()

plt.plot(list(C_d1.keys()), list(C_d1.values()), label = 'Cd1')
plt.plot(list(C_d2.keys()), list(C_d2.values()), label = 'Cd2')
plt.xlabel('Orientation of rotor arm')
plt.ylabel('Drag coefficient [-]')
plt.suptitle('Drag coefficient of blades during rotation due to the angle of attack')
plt.legend( bbox_to_anchor=(1.3, 0.5), loc='center right')
plt.show()

plt.plot(list(C_l1.keys()), list(C_l1.values()), label = 'Cl1')
plt.plot(list(C_l2.keys()), list(C_l2.values()), label = 'Cl2')
plt.xlabel('Orientation of rotor arm')
plt.ylabel('Lift coefficient [-]')
plt.suptitle('Lift coefficient of blades during rotation')
plt.legend( bbox_to_anchor=(1.3, 0.5), loc='center right')
plt.show()

plt.plot(list(M_Tangent_Tot.keys()), list(M_Tangent_Tot.values()))
plt.xlabel('Orientation of rotor arm')
plt.ylabel('Momentum [Nm]')
plt.suptitle('Rotational momentum of all arms combined')
plt.show()


mydata = [
    ["Maximal force combination on outer blade",F_1_Max], 
    ["Maximal force combination on outer blade", F_2_Max], 
    ["Maximal tention on rotor arm",F_P_MaxTot], 
    ["Maximal compression on rotor arm",F_P_MinTot],
    ['Maximum turning momentum at center',M_Center_MAX],
    ['Maximum counter turning momentum at center',M_Center_MIN]
]
 
# create header
head = ["Force/Moment", "[Deg, N/Nm"]
 
# display table
print(tabulate(mydata, headers=head, tablefmt="grid"))














