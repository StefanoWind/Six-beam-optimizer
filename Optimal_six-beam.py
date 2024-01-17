# -*- coding: utf-8 -*-
#03/11/2022: created
#03/14/2022: finalized
#12/22/2023: created, finalized
#01/16/2024: adapted for WFIP3

import numpy as np
import matplotlib.pyplot as plt
import utils as utl
from scipy.optimize import minimize
import matplotlib

plt.close('all')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] =14

#%% Inputs

#constraints
min_beta=60#[deg] minimum elevation angle
xmin=-1000#[m] distance of the non-homogeneous region from the lidar
zmax=300#[m] height of the non-homogenoeus region (only if xmin<0)
zmin=100#[m] minimum height for profiling (only if xmin>0)

#optimization
tolerance=1e-7 #convergence criterion
max_iter=1000# maximum iteration
N_opt=20 #number of optimizations with random intial seed

#graphics
inf=1000#[m]

#%% Functions
def err_RS(x):
#error propgation of variance of single LOS into Reynolds Stresses (Sathe at al, 2016)
#12/22/2023: finalized

    a=np.array(x[6:])
    b=np.array(x[:6])
    N=np.array([[7/8,1/8,0,0,0,0],
                [1/8,7/8,0,0,0,0],
                [0,0,1,0,0,0],
                [0,0,0,3/2,0,0],
                [0,0,0,0,1,0],
                [0,0,0,0,0,1]])
    
    sa=utl.sind(a)
    ca=utl.cosd(a)
    sb=utl.sind(b)
    cb=utl.cosd(b)
   
    M=np.zeros((6,6))
    
    M[:,0]=cb**2*ca**2
    M[:,1]=cb**2*sa**2
    M[:,2]=sb**2
    M[:,3]=2*cb**2*ca*sa
    M[:,4]=2*cb*sb*ca
    M[:,5]=2*cb*sb*sa
    
    if np.linalg.det(M)!=0:
        M_inv=np.linalg.inv(M)
        err=np.trace(np.matmul(N,np.matmul(M_inv,np.transpose(M_inv))))
    else:
        err=np.inf
    
    return err

#%% Initialization
f_opt=[]
beta_opt=[]
alpha_opt=[]

f_best=[]
beta_best=[]
alpha_best=[]

if xmin<=0:
    z0=zmax
else:
    z0=zmin
    
#%% Main
while len(f_opt)<N_opt:
    
    if xmin<=0:
        cons=({'type': 'ineq', 'fun': lambda x: min(1/utl.tand(x[:6])*utl.cosd(x[6:]))-xmin/zmax})
        bous=((min_beta,90),(min_beta,90),(min_beta,90),(min_beta,90),(min_beta,90),(min_beta,90),
              (0,360),(0,360),(0,360),(0,360),(0,360),(0,360))
    else:
        max_beta=utl.arctand(zmin/xmin)
        cons=({'type': 'ineq', 'fun': lambda x: min(1/utl.tand(x[:6])*utl.cosd(x[6:]))-xmin/zmin})
        bous=((min_beta,max_beta),(min_beta,max_beta),(min_beta,max_beta),(min_beta,max_beta),(min_beta,max_beta),(min_beta,max_beta),
              (-90,90),(-90,90),(-90,90),(-90,90),(-90,90),(-90,90))
        
    beta0= [np.random.randint(bous[i][0],bous[i][1]) for i in range(6)]
    alpha0=[np.random.randint(bous[i][0],bous[i][1]) for i in range(6,12)]
     
    res = minimize(err_RS, beta0+alpha0, method='SLSQP', tol=tolerance,
                   bounds=bous,
                   constraints=cons,options={'maxiter':max_iter})
    
    if res.success==True:       
        beta_opt.append(np.around(res.x[:6],2))
        alpha_opt.append(np.around(res.x[6:],2))
        f_opt.append(np.around(res.fun,2))
    
        f_best.append(min(f_opt))
        beta_best.append(beta_opt[np.where(f_opt==f_best[-1])[0][0]])
        alpha_best.append(alpha_opt[np.where(f_opt==f_best[-1])[0][0]])
    else:
        print(res.message)
    print(str(len(f_opt))+' out of '+str(N_opt)+' optimizations completed')


#%% Plot
fig=plt.figure(figsize=(20,8))
ax = fig.add_subplot(1,2,1, projection='3d')
for b,a in zip(beta_best[-1],alpha_best[-1]):           
    r=inf
    x=np.append(0,utl.cosd(b)*utl.cosd(a)*r)
    y=np.append(0,utl.cosd(b)*utl.sind(a)*r)
    z=np.append(0,utl.sind(b)*r)
    ax.plot3D(x,y,z,'r',alpha=0.5)

if xmin<=0:
    plt.title(r'$z_{max}='+str(z0)+'$ m, $x_{min}='+str(xmin)+'$ m ,'+'$N_{0py}='+str(N_opt)+'$, $\sigma_{RS}^2/\sigma_{LOS}^2=$'+str(f_best[-1])+'\n'+
              r'$\beta=['+str(utl.vec2str(np.round(beta_best[-1],1),', ','%06.2f'))+']^\circ$'+'\n'+
              r'$\alpha=['+str(utl.vec2str(np.round(alpha_best[-1],1),', ','%06.2f'))+']^\circ$')
if xmin>0:
    plt.title(r'$z_{min}='+str(z0)+'$ m, $x_{min}='+str(xmin)+'$ m ,'+'$N_{0py}='+str(N_opt)+'$, $\sigma_{RS}^2/\sigma_{LOS}^2=$'+str(f_best[-1])+'\n'+
              r'$\beta=['+str(utl.vec2str(np.round(beta_best[-1],1),', ','%06.2f'))+']^\circ$'+'\n'+
              r'$\alpha=['+str(utl.vec2str(np.round(alpha_best[-1],1),', ','%06.2f'))+']^\circ$')
    
plt.xlabel(r'$x$ [m]')
plt.ylabel(r'$y$ [m]')
ax.set_zlabel('$z$ [m]')
if xmin<0:
    vertices = [[-inf, -inf, 0],
                [xmin, -inf, 0],
                [xmin, inf, 0],
                [-inf, inf, 0],
                [-inf, -inf, zmax],
                [xmin, -inf, zmax],
                [xmin, inf, zmax],
                [-inf, inf, zmax]]
    
else:
    vertices = [[-inf, -inf, 0],
                [xmin, -inf, 0],
                [xmin, inf, 0],
                [-inf, inf, 0],
                [-inf, -inf, inf],
                [xmin, -inf, inf],
                [xmin, inf, inf],
                [-inf, inf, inf]]
    
utl.draw_cube(ax,vertices)
if xmin>0:
    vertices = [[xmin, -inf, 0],
                [inf, -inf, 0],
                [inf, inf, 0],
                [xmin, inf, 0],
                [xmin, -inf, zmin],
                [inf, -inf, zmin],
                [inf, inf, zmin],
                [xmin, inf, zmin]]
    utl.draw_cube(ax,vertices)
    


plt.grid()
ax.view_init(elev=35, azim=-75)
plt.xlim([-inf,inf])
plt.ylim([-inf,inf])
ax.set_zlim([0,inf])
utl.axis_equal()

plt.subplot(1,2,2)   
plt.plot(np.arange(1,N_opt+1),np.array(f_best),'.-k')
plt.xticks(np.arange(1,N_opt+1))
plt.xlabel('Number of optimizations')
plt.ylabel( 'Best $\sigma_{RS}^2/\sigma_{LOS}^2$')
plt.yscale('log')
plt.tight_layout()
plt.title('Objective function evolution')
plt.grid()
