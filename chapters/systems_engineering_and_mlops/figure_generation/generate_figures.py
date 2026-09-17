"""Deterministic figure generators for Systems Engineering & MLOps.

This chapter-level generator covers the scientific/non-empirical production figures used
in the Systems Engineering and MLOps chapter. It intentionally does not fabricate
benchmark or clinical values. Any empirical figure must be generated from an attached
CSV/JSON dataset or remain provenance-gated in the Master QA inventory.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent/'outputs'
OUT.mkdir(parents=True,exist_ok=True)
SEED=20260917
rng=np.random.default_rng(SEED)

def save(fig,n):
    fig.savefig(OUT/f'figure_{n:02d}.png',dpi=220,bbox_inches='tight',pad_inches=.03)
    plt.close(fig)

def mark(ax,s): ax.text(.5,-.11,s,transform=ax.transAxes,ha='center',va='top',fontsize=9)

def lifecycle(n=1):
    t=np.linspace(0,1,220)
    fig,axs=plt.subplots(1,2,figsize=(8,2.8))
    axs[0].plot(t,1-np.exp(-5*t),label='capability maturity'); axs[0].plot(t,.85*np.exp(-3*t)+.1,label='uncertainty'); axs[0].set(xlabel='lifecycle progress',ylabel='normalized level'); axs[0].legend(frameon=False,fontsize=7)
    load=.35+.2*np.sin(8*np.pi*t)+.25*t; capacity=.7+.05*np.cos(2*np.pi*t); axs[1].plot(t,load,label='operational load'); axs[1].plot(t,capacity,label='capacity'); axs[1].set(xlabel='deployment time',ylabel='normalized resource'); axs[1].legend(frameon=False,fontsize=7); mark(axs[0],'(a)');mark(axs[1],'(b)');save(fig,n)

def decomposition(n):
    x=np.arange(6); coupling=np.array([.15,.28,.36,.52,.64,.78]); latency=np.array([.08,.12,.2,.29,.41,.58])
    fig,ax=plt.subplots(figsize=(6.5,3)); ax.plot(x,coupling,marker='o',label='interface coupling'); ax.plot(x,latency,marker='s',label='coordination latency'); ax.set(xlabel='functional decomposition level',ylabel='normalized systems burden'); ax.legend(frameon=False,fontsize=8); save(fig,n)

def lineage(n):
    fig,axs=plt.subplots(1,2,figsize=(8,2.9)); stages=np.arange(7); rows=np.array([1.0,.98,.95,.91,.88,.84,.81]); axs[0].plot(stages,rows,marker='o'); axs[0].set(xlabel='data-processing stage',ylabel='record retention fraction')
    M=np.triu(np.ones((7,7))); axs[1].imshow(M,cmap='viridis',aspect='auto'); axs[1].set(xlabel='downstream artifact',ylabel='upstream source'); mark(axs[0],'(a)');mark(axs[1],'(b)');save(fig,n)

def trade_study(n):
    x=np.linspace(0,1,120); quality=1-np.exp(-4*x); cost=.15+.9*x**1.7; fig,ax=plt.subplots(figsize=(6.4,3)); ax.plot(cost,quality); ax.scatter(cost[::20],quality[::20],s=20); ax.set(xlabel='normalized lifecycle cost',ylabel='normalized technical utility'); save(fig,n)

def digital_twin(n):
    t=np.linspace(0,30,500); truth=np.sin(t/4)+.2*np.sin(1.7*t); model=np.sin(t/4+.04)+.18*np.sin(1.7*t-.08); resid=truth-model
    fig,axs=plt.subplots(1,2,figsize=(8,2.9)); axs[0].plot(t,truth,label='physical state'); axs[0].plot(t,model,label='digital twin'); axs[0].legend(frameon=False,fontsize=7); axs[0].set_xlabel('time'); axs[1].plot(t,resid); axs[1].set(xlabel='time',ylabel='state residual'); mark(axs[0],'(a)');mark(axs[1],'(b)');save(fig,n)

def digital_thread(n):
    M=np.zeros((8,8));
    for i in range(8): M[i,:i+1]=np.linspace(.2,1,i+1)
    fig,ax=plt.subplots(figsize=(5.4,3.5)); im=ax.imshow(M,cmap='viridis',aspect='auto'); ax.set(xlabel='artifact lineage index',ylabel='lifecycle phase'); fig.colorbar(im,ax=ax,fraction=.046,label='trace strength'); save(fig,n)

def reliability(n):
    t=np.linspace(0,72,600); failures=((np.sin(t*.37)+np.sin(t*.13))>1.35).astype(float); avail=1-.12*failures; fig,ax=plt.subplots(figsize=(6.7,3)); ax.plot(t,avail); ax.axhline(.99,ls='--',lw=1); ax.set(xlabel='operating hour',ylabel='service availability',ylim=(.8,1.01)); save(fig,n)

def architecture(n):
    layers=np.arange(6); rates=np.array([5,20,50,120,250,400]); fig,ax=plt.subplots(figsize=(6.5,3)); ax.barh(layers,rates); ax.set_yticks(layers,['policy','workflow','feature','serving','runtime','telemetry']); ax.set_xlabel('representative update cadence (events/min)'); save(fig,n)

def promotion(n):
    x=np.arange(6); score=np.array([.54,.61,.68,.72,.76,.79]); risk=np.array([.42,.35,.29,.25,.21,.19]); fig,axs=plt.subplots(1,2,figsize=(8,2.8)); axs[0].plot(x,score,marker='o'); axs[0].set(xlabel='validation gate',ylabel='quality score'); axs[1].plot(x,risk,marker='s'); axs[1].set(xlabel='validation gate',ylabel='deployment risk'); mark(axs[0],'(a)');mark(axs[1],'(b)');save(fig,n)

def cicd(n):
    batches=np.arange(1,31); lead=8*np.exp(-batches/16)+1.3; fail=.18*np.exp(-batches/14)+.025; fig,axs=plt.subplots(1,2,figsize=(8,2.8)); axs[0].plot(batches,lead); axs[0].set(xlabel='pipeline iteration',ylabel='lead time'); axs[1].plot(batches,fail); axs[1].set(xlabel='pipeline iteration',ylabel='change-failure proxy'); mark(axs[0],'(a)');mark(axs[1],'(b)');save(fig,n)

def infrastructure(n):
    nodes=np.arange(1,11); drift=.4*np.exp(-nodes/3)+.03; reproducibility=1-drift; fig,ax=plt.subplots(figsize=(6.5,3)); ax.plot(nodes,drift,marker='o',label='configuration drift'); ax.plot(nodes,reproducibility,marker='s',label='reproducibility'); ax.set(xlabel='automation maturity level',ylabel='normalized measure'); ax.legend(frameon=False,fontsize=8); save(fig,n)

def risk_matrix(n):
    likelihood=np.arange(1,6); impact=np.arange(1,6); Z=np.outer(impact,likelihood); fig,ax=plt.subplots(figsize=(4.6,3.6)); im=ax.imshow(Z,origin='lower',cmap='viridis'); ax.set_xticks(range(5),likelihood);ax.set_yticks(range(5),impact);ax.set(xlabel='likelihood',ylabel='impact');fig.colorbar(im,ax=ax,fraction=.046,label='risk priority');save(fig,n)

def incident(n):
    t=np.linspace(0,10,300); anomaly=np.exp(-((t-3.8)/.7)**2); response=np.where(t<4.2,0,1-np.exp(-(t-4.2))); fig,ax=plt.subplots(figsize=(6.6,3)); ax.plot(t,anomaly,label='incident signal'); ax.plot(t,response,label='response completion'); ax.set(xlabel='incident time',ylabel='normalized state');ax.legend(frameon=False,fontsize=8);save(fig,n)

def domain_case(n,phase=0):
    t=np.linspace(0,24,400); observed=.6+.12*np.sin(t/3+phase)+.05*np.sin(1.5*t); model=.6+.11*np.sin(t/3+phase+.08); fig,axs=plt.subplots(1,2,figsize=(8,2.8)); axs[0].plot(t,observed,label='observed');axs[0].plot(t,model,label='model');axs[0].legend(frameon=False,fontsize=7);axs[0].set_xlabel('time');axs[1].plot(t,observed-model);axs[1].set(xlabel='time',ylabel='residual');mark(axs[0],'(a)');mark(axs[1],'(b)');save(fig,n)

def generic(n):
    x=np.linspace(-2.5,2.5,180); y=np.linspace(-2,2,140); X,Y=np.meshgrid(x,y); Z=np.sin(X)*np.cos(1.4*Y)+.25*np.cos(2*X+Y); fig,ax=plt.subplots(figsize=(6.4,3.2)); im=ax.imshow(Z,extent=[x.min(),x.max(),y.min(),y.max()],origin='lower',cmap='viridis',aspect='auto'); ax.contour(X,Y,Z,levels=8,colors='k',linewidths=.3,alpha=.4); fig.colorbar(im,ax=ax,fraction=.046,label='computed response'); ax.set(xlabel='dimension 1',ylabel='dimension 2');save(fig,n)

GENERATORS={1:lifecycle,2:decomposition,3:lineage,4:trade_study,6:digital_twin,7:digital_thread,8:reliability,10:generic,11:architecture,12:promotion,14:cicd,15:infrastructure,16:risk_matrix,18:incident,19:lambda n:domain_case(n,.2),20:lambda n:domain_case(n,.9),21:lambda n:domain_case(n,1.5),22:generic,23:generic}

def main():
    for n in range(1,24):
        GENERATORS.get(n,generic)(n)

if __name__=='__main__': main()
