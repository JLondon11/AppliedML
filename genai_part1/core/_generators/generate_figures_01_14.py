from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

NAVY="#173A52"; STEEL="#6C8796"; TEAL="#5F7E7A"; VIOLET="#75637A"; ORANGE="#C8753D"
GRAPHITE="#30343B"; LIGHT="#D9E0E5"; PALE="#EEF2F4"
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.labelsize":9,
                     "figure.facecolor":"white","axes.facecolor":"white",
                     "savefig.facecolor":"white","mathtext.fontset":"dejavuserif"})

ROOT=Path(__file__).resolve().parents[2]
PATHS={
1:"01_introduction_chapter_roadmap",2:"02_representation_learning_latent_spaces",
3:"03_variational_autoencoders",4:"04_conditional_hierarchical_vaes",
5:"05_vector_quantized_models",6:"06_deep_convolutional_gans",
7:"07_conditional_gans_image_translation",8:"08_wasserstein_stylegan_evaluation",
9:"09_diffusion_foundations",10:"10_ddpm_ddim_score_models",
11:"11_latent_diffusion",12:"12_stable_diffusion_architecture",
13:"13_stable_diffusion_cross_attention",14:"14_controlnet_structured_conditioning"}

def outdir(n):
    p=ROOT/"core"/PATHS[n]/"figures"/f"figure{n:02d}"/"outputs"
    p.mkdir(parents=True,exist_ok=True); return p
def save(fig,n):
    p=outdir(n)
    fig.savefig(p/f"figure{n:02d}.svg",bbox_inches="tight")
    fig.savefig(p/f"figure{n:02d}.pdf",bbox_inches="tight")
    fig.savefig(p/f"figure{n:02d}.png",dpi=300,bbox_inches="tight")
    plt.close(fig)
def clean(ax):
    for s in ax.spines.values(): s.set_linewidth(.55)
def g2(X,Y,mx,my,sx,sy,a=1):
    return a*np.exp(-.5*(((X-mx)/sx)**2+((Y-my)/sy)**2))

def figure01():
    fig=plt.figure(figsize=(10.6,6.8)); gs=fig.add_gridspec(2,3,hspace=.36,wspace=.28)
    x=np.linspace(-3,3,110); y=np.linspace(-3,3,110); X,Y=np.meshgrid(x,y)
    Z=g2(X,Y,-1.3,.7,.7,.8)+g2(X,Y,.7,1.1,.6,.8,.85)+g2(X,Y,.8,-1,.8,.6,.7)
    ax=fig.add_subplot(gs[0,0],projection="3d"); Z0=g2(X,Y,0,0,1,1)
    ax.plot_surface(X,Y,Z0,cmap="Blues",linewidth=0,alpha=.94); ax.plot_surface(X,Y,Z,cmap="Oranges",linewidth=0,alpha=.72)
    ax.set(xlabel="$z_1$",ylabel="$z_2$",zlabel="$p$"); ax.view_init(28,-58); ax.text2D(.05,.9,r"$p(z)\rightarrow p_\theta(x)$",transform=ax.transAxes)
    ax=fig.add_subplot(gs[1,0]); ax.contour(X,Y,Z,6,colors=[NAVY],linewidths=.9)
    Zg=g2(X,Y,-1,.5,.72,.78,.9)+g2(X,Y,1,-.7,.7,.7,.65); ax.contour(X,Y,Zg,6,colors=[ORANGE],linewidths=.9,linestyles="--")
    ax.set(xlabel="$x_1$",ylabel="$x_2$"); ax.set_aspect("equal"); clean(ax)
    ax=fig.add_subplot(gs[0,1],projection="3d"); ax.plot_surface(X,Y,Z,cmap="Blues",linewidth=0,alpha=.97); ax.contour(X,Y,Z,zdir="z",offset=0,cmap="Blues",levels=7,linewidths=.6)
    ax.set(xlabel="$x_1$",ylabel="$x_2$",zlabel="$p(x)$"); ax.view_init(30,-58); ax.text2D(.35,.92,r"$p_{\rm data}(x)$",transform=ax.transAxes,fontsize=11)
    ax=fig.add_subplot(gs[1,1]); u=np.linspace(-2.3,2.3,12)
    for c in u:
        yy=np.linspace(-2.3,2.3,150); xx=np.full_like(yy,c); ax.plot(xx+.55*np.sin(yy)*np.exp(-.12*(xx**2+yy**2)),yy+.45*np.sin(xx)*np.exp(-.12*(xx**2+yy**2)),color=STEEL,lw=.65)
    for c in u:
        xx=np.linspace(-2.3,2.3,150); yy=np.full_like(xx,c); ax.plot(xx+.55*np.sin(yy)*np.exp(-.12*(xx**2+yy**2)),yy+.45*np.sin(xx)*np.exp(-.12*(xx**2+yy**2)),color=STEEL,lw=.65)
    ax.set(xlabel=r"$z_1\mapsto x_1$",ylabel=r"$z_2\mapsto x_2$"); ax.set_aspect("equal"); clean(ax)
    ax=fig.add_subplot(gs[0,2],projection="3d"); xi=np.linspace(-3,3,180)
    for i,m in enumerate([-1.5,-.7,.15,.9,1.5]): ax.plot(xi,np.full_like(xi,i),np.exp(-.5*((xi-m)/(.55+.06*i))**2),color=NAVY,lw=1.1)
    ax.set(xlabel="$x_i$",ylabel="$i$",zlabel=r"$p(x_i|x_{<i})$"); ax.view_init(25,-60)
    ax=fig.add_subplot(gs[1,2],projection="3d")
    for j,t in enumerate([1,.72,.48,.25,0]):
        zz=(1-t)*Z+t*g2(X,Y,0,0,1.6,1.6); ax.plot(x,np.full_like(x,j),zz[len(y)//2,:],color=[STEEL,STEEL,TEAL,VIOLET,ORANGE][j],lw=1.15)
    ax.set(xlabel="$x$",ylabel="$t$",zlabel="$p_t(x)$"); ax.view_init(26,-60)
    labs=["Foundations","Model families","Evaluation","Compute","Applications","Case studies"]; xx=[.12,.30,.49,.66,.81,.94]
    for p,l in zip(xx,labs): fig.text(p,.02,l,ha="center",color=GRAPHITE)
    for a,b in zip(xx[:-1],xx[1:]): fig.add_artist(FancyArrowPatch((a+.04,.022),(b-.05,.022),transform=fig.transFigure,arrowstyle="->",mutation_scale=8,lw=.7,color=GRAPHITE))
    save(fig,1)

def figure02():
    fig=plt.figure(figsize=(8.3,4.5)); ax=fig.add_subplot(1,2,1,projection="3d")
    u=np.linspace(-2.2,2.2,90); v=np.linspace(-1.2,1.2,34); U,V=np.meshgrid(u,v)
    X=U;Y=np.sin(U)*1.2+V*np.cos(U*.8);Z=np.cos(U)*.9+V*np.sin(U*.8)
    ax.plot_surface(X,Y,Z,color=LIGHT,edgecolor=STEEL,linewidth=.22,alpha=.92)
    t=np.linspace(-1.8,1.8,100);ax.plot(t,np.sin(t)*1.2,np.cos(t)*.9,color=ORANGE,lw=2.2); ax.scatter(t[::15],np.sin(t[::15])*1.2,np.cos(t[::15])*.9,s=10,color=NAVY)
    ax.set(xlabel="$x_1$",ylabel="$x_2$",zlabel="$x_3$"); ax.view_init(22,-55)
    ax=fig.add_subplot(1,2,2); z=np.linspace(-2,2,200);ax.plot(z,np.sin(1.1*z),color=NAVY,lw=1.7);ax.scatter(z[::22],np.sin(1.1*z[::22]),s=13,color=ORANGE)
    ax.set(xlabel="Latent coordinate $z$",ylabel="Semantic coordinate");ax.grid(True,lw=.4,alpha=.25);clean(ax);save(fig,2)

def figure03():
    fig,ax=plt.subplots(figsize=(9.2,3.5)); ax.set_axis_off(); xs=np.linspace(-2.5,2.5,200)
    ax.plot(.08+.15*(xs-xs.min())/(xs.max()-xs.min()),.5+.18*np.exp(-.5*(xs/.85)**2),color=NAVY,lw=1.3,transform=ax.transAxes);ax.text(.14,.77,"$x$",transform=ax.transAxes,ha="center")
    ax.add_patch(Rectangle((.28,.42),.13,.20,transform=ax.transAxes,fill=False,lw=1,ec=GRAPHITE));ax.text(.345,.52,r"$\mu_\phi(x),\,\log\sigma_\phi^2(x)$",transform=ax.transAxes,ha="center",va="center")
    t=np.linspace(-2.7,2.7,220); q=np.exp(-.5*((t-.2)/.75)**2);ax.plot(.46+.10*(t-t.min())/(t.max()-t.min()),.43+.18*q,transform=ax.transAxes,color=VIOLET,lw=1.4);ax.text(.51,.68,r"$q_\phi(z|x)$",transform=ax.transAxes,ha="center");ax.text(.51,.35,r"$z=\mu+\sigma\odot\epsilon$",transform=ax.transAxes,ha="center")
    qq=np.exp(-.5*((xs+.3)/.9)**2);ax.plot(.70+.12*(xs-xs.min())/(xs.max()-xs.min()),.43+.18*qq,transform=ax.transAxes,color=ORANGE,lw=1.4);ax.text(.76,.68,r"$p_\theta(x|z)$",transform=ax.transAxes,ha="center")
    ax.add_patch(Rectangle((.86,.42),.08,.20,transform=ax.transAxes,fill=False,lw=1,ec=GRAPHITE));ax.text(.90,.52,r"$\hat{x}$",transform=ax.transAxes,ha="center",va="center")
    for a,b in [(.23,.28),(.41,.46),(.56,.70),(.82,.86)]:ax.add_artist(FancyArrowPatch((a,.52),(b,.52),transform=ax.transAxes,arrowstyle="->",mutation_scale=11,lw=.9,color=GRAPHITE))
    ax.text(.51,.18,r"$\mathcal{L}=\mathbb{E}_{q_\phi}[\log p_\theta(x|z)]-D_{\mathrm{KL}}(q_\phi(z|x)\|p(z))$",transform=ax.transAxes,ha="center",fontsize=10);save(fig,3)

def figure04():
    fig,axs=plt.subplots(1,2,figsize=(8.6,4));[a.set_axis_off() for a in axs]
    ax=axs[0];pts={"x":(.18,.55),"c":(.18,.25),"z":(.52,.55),"xh":(.84,.55)}
    for k,(x,y) in pts.items():
        ax.add_patch(Circle((x,y),.07,transform=ax.transAxes,facecolor=PALE if k in ("x","c") else "white",edgecolor=GRAPHITE,lw=1));ax.text(x,y,{"x":"$x$","c":"$c$","z":"$z$","xh":"$x'$"}[k],transform=ax.transAxes,ha="center",va="center")
    for a,b in [("x","z"),("c","z"),("z","xh"),("c","xh")]:
        x1,y1=pts[a];x2,y2=pts[b];ax.add_artist(FancyArrowPatch((x1+.07,y1),(x2-.07,y2),transform=ax.transAxes,arrowstyle="->",mutation_scale=10,lw=.9,color=NAVY))
    ax=axs[1];levels=[(.18,.72),(.42,.58),(.63,.43),(.82,.28)]
    for (x,y),lab in zip(levels,["$z_3$","$z_2$","$z_1$","$x$"]):ax.add_patch(Circle((x,y),.065,transform=ax.transAxes,facecolor="white",edgecolor=GRAPHITE,lw=1));ax.text(x,y,lab,transform=ax.transAxes,ha="center",va="center")
    for (x1,y1),(x2,y2) in zip(levels[:-1],levels[1:]):ax.add_artist(FancyArrowPatch((x1+.06,y1-.01),(x2-.06,y2+.01),transform=ax.transAxes,arrowstyle="->",mutation_scale=10,lw=.9,color=VIOLET))
    save(fig,4)

def figure05():
    rng=np.random.default_rng(5);fig,axs=plt.subplots(1,2,figsize=(8.3,3.9));code=np.array([[-1.7,-1.2],[-.7,1],[.3,-.6],[1.2,1.3],[1.7,-.3]]);pts=np.vstack([c+rng.normal(scale=.34,size=(40,2)) for c in code]);lab=((pts[:,None,:]-code[None,:,:])**2).sum(-1).argmin(1)
    ax=axs[0];ax.scatter(pts[:,0],pts[:,1],s=8,c=[STEEL],alpha=.45);ax.scatter(code[:,0],code[:,1],s=70,c=[NAVY],marker="D",edgecolor="white",lw=.7)
    for i in range(0,len(pts),12):ax.add_artist(FancyArrowPatch(tuple(pts[i]),tuple(code[lab[i]]),arrowstyle="->",mutation_scale=8,lw=.55,color=ORANGE,alpha=.7))
    ax.set(xlabel="$z_{e,1}$",ylabel="$z_{e,2}$");ax.set_aspect("equal");ax.grid(True,lw=.35,alpha=.2);clean(ax)
    ax=axs[1];cnt=np.bincount(lab,minlength=5);ax.bar(np.arange(5),cnt,width=.65,color=STEEL,edgecolor=NAVY,lw=.6);ax.set(xlabel="Codebook index",ylabel="Assignments");ax.set_xticks(np.arange(5));ax.grid(axis="y",lw=.35,alpha=.2);clean(ax);save(fig,5)

def stack(ax,x,y,w,h,n,dx,dy,col):
    for i in range(n):ax.add_patch(Rectangle((x+i*dx,y+i*dy),w,h,transform=ax.transAxes,facecolor="none",edgecolor=col,lw=.75))
def figure06():
    fig,ax=plt.subplots(figsize=(9.2,4));ax.set_axis_off();xs=[.08,.22,.38,.56,.76];wh=[(.05,.14),(.07,.18),(.10,.23),(.13,.28),(.17,.32)]
    for i,(x,(w,h)) in enumerate(zip(xs,wh)):
        stack(ax,x,.59,w,h,4,.008,.008,NAVY)
        if i<4:ax.add_artist(FancyArrowPatch((x+w+.04,.72),(xs[i+1]-.01,.72),transform=ax.transAxes,arrowstyle="->",mutation_scale=10,lw=.9,color=GRAPHITE))
    xs2=[.08,.26,.45,.64,.82];wh2=[(.17,.32),(.13,.28),(.10,.23),(.07,.18),(.045,.14)]
    for i,(x,(w,h)) in enumerate(zip(xs2,wh2)):
        stack(ax,x,.11,w,h,4,.008,.008,ORANGE)
        if i<4:ax.add_artist(FancyArrowPatch((x+w+.04,.27),(xs2[i+1]-.01,.27),transform=ax.transAxes,arrowstyle="->",mutation_scale=10,lw=.9,color=GRAPHITE))
    ax.text(.06,.83,"$z$",transform=ax.transAxes);ax.text(.83,.75,"$x_G$",transform=ax.transAxes);ax.text(.06,.16,"$x$",transform=ax.transAxes);ax.text(.88,.23,"$D(x)$",transform=ax.transAxes);save(fig,6)

def figure07():
    fig,axs=plt.subplots(1,2,figsize=(8.7,4.1));t=np.linspace(0,2*np.pi,120);A=np.c_[np.cos(t),.55*np.sin(t)];B=np.c_[1.9+.9*np.cos(t),.15+.55*np.sin(t)]
    ax=axs[0];ax.plot(A[:,0],A[:,1],color=NAVY,lw=1.2);ax.plot(B[:,0],B[:,1],color=ORANGE,lw=1.2)
    for k in range(0,120,20):ax.add_artist(FancyArrowPatch(tuple(A[k]),tuple(B[k]),arrowstyle="->",mutation_scale=9,lw=.7,color=GRAPHITE))
    ax.set_aspect("equal");ax.grid(True,lw=.35,alpha=.18);clean(ax)
    ax=axs[1];ax.plot(A[:,0],A[:,1],color=NAVY,lw=1.2);ax.plot(B[:,0],B[:,1],color=ORANGE,lw=1.2);p=A[18];q=B[43];ax.add_artist(FancyArrowPatch(tuple(p),tuple(q),connectionstyle="arc3,rad=.16",arrowstyle="->",mutation_scale=10,lw=1,color=VIOLET));ax.add_artist(FancyArrowPatch(tuple(q),tuple(p),connectionstyle="arc3,rad=.16",arrowstyle="->",mutation_scale=10,lw=1,color=TEAL));ax.text(.1,.9,r"$F(G(x))\approx x$",transform=ax.transAxes);ax.set_aspect("equal");ax.grid(True,lw=.35,alpha=.18);clean(ax);save(fig,7)

def figure08():
    fig,axs=plt.subplots(1,3,figsize=(9.4,3.5));x=np.linspace(-3,3,300);p=np.exp(-.5*((x+1)/.75)**2);q=np.exp(-.5*((x-1)/.8)**2)
    axs[0].plot(x,p,color=NAVY,lw=1.3,label="$p_{data}$");axs[0].plot(x,q,color=ORANGE,lw=1.3,label="$p_G$");axs[0].plot(x,.55*np.tanh(.8*x)+.15,color=VIOLET,lw=1.1,label="$f_w$");axs[0].legend(frameon=False,fontsize=7);axs[0].grid(True,lw=.35,alpha=.2);clean(axs[0])
    r=np.linspace(0,1,200)
    for i,(f,c) in enumerate([(1,NAVY),(3,TEAL),(7,VIOLET),(12,ORANGE)]):axs[1].plot(r,.22*i+.08*np.sin(2*np.pi*f*r)*np.exp(-1.1*r),color=c,lw=1.1)
    axs[1].set_yticks([]);axs[1].grid(True,lw=.35,alpha=.2);clean(axs[1])
    pts=np.array([[.78,.43],[.61,.72],[.83,.70],[.46,.84]]);axs[2].scatter(pts[:,0],pts[:,1],s=[60,60,80,55],c=[NAVY,TEAL,VIOLET,ORANGE],edgecolor="white",lw=.7);axs[2].set(xlim=(0,1),ylim=(0,1),xlabel="precision / fidelity",ylabel="recall / coverage");axs[2].grid(True,lw=.35,alpha=.2);clean(axs[2]);save(fig,8)

def figure09():
    fig,ax=plt.subplots(figsize=(8.8,3.8));x=np.linspace(-4,4,500);base=.65*np.exp(-.5*((x+1.2)/.55)**2)+.45*np.exp(-.5*((x-1.3)/.75)**2)
    for i,t in enumerate(np.linspace(0,1,5)):
        sig=.3+1.5*t;z=(1-t)*base+t*np.exp(-.5*(x/sig)**2);z/=z.max();ax.plot(x,z+i*1.05,color=[ORANGE,VIOLET,TEAL,STEEL,NAVY][i],lw=1.5)
        if i<4:ax.add_artist(FancyArrowPatch((2.8,i*1.05+.45),(2.8,(i+1)*1.05+.45),arrowstyle="->",mutation_scale=10,lw=.8,color=GRAPHITE))
    ax.set_yticks(np.arange(5)*1.05,["$x_0$","$x_{t_1}$","$x_{t_2}$","$x_{t_3}$","$x_T$"]);ax.set_xlabel("$x$");ax.grid(axis="x",lw=.35,alpha=.18);clean(ax);save(fig,9)

def figure10():
    rng=np.random.default_rng(10);fig,ax=plt.subplots(figsize=(6.7,5.2));th=np.linspace(0,2*np.pi,240);ax.plot(2*np.cos(th),1.25*np.sin(th),color=LIGHT,lw=1);gx=np.linspace(-3,3,17);gy=np.linspace(-2.4,2.4,15);GX,GY=np.meshgrid(gx,gy);R=np.sqrt((GX/2)**2+(GY/1.25)**2)+1e-6;U=-(R-1)*GX/(R*2);V=-(R-1)*GY/(R*1.25);ax.quiver(GX,GY,U,V,color=STEEL,alpha=.5,scale=18,width=.0025)
    p=np.array([2.7,2.]);ps=[p.copy()]
    for _ in range(14):r=np.sqrt((p[0]/2)**2+(p[1]/1.25)**2)+1e-6;p=p+.35*np.array([-(r-1)*p[0]/(r*2),-(r-1)*p[1]/(r*1.25)])+rng.normal(scale=.08,size=2);ps.append(p.copy())
    ps=np.array(ps);ax.plot(ps[:,0],ps[:,1],"o-",ms=2.8,lw=1,color=VIOLET,label="stochastic reverse")
    p=np.array([2.7,2.]);pd=[p.copy()]
    for _ in range(8):r=np.sqrt((p[0]/2)**2+(p[1]/1.25)**2)+1e-6;p=p+.55*np.array([-(r-1)*p[0]/(r*2),-(r-1)*p[1]/(r*1.25)]);pd.append(p.copy())
    pd=np.array(pd);ax.plot(pd[:,0],pd[:,1],"s--",ms=3,lw=1.2,color=ORANGE,label="deterministic / accelerated");ax.legend(frameon=False,fontsize=8);ax.set(xlabel="$x_1$",ylabel="$x_2$");ax.set_aspect("equal");ax.grid(True,lw=.35,alpha=.2);clean(ax);save(fig,10)

def figure11():
    fig,ax=plt.subplots(figsize=(9.5,3.8));ax.set_axis_off();ax.add_patch(Rectangle((.05,.36),.13,.28,transform=ax.transAxes,fill=False,ec=NAVY,lw=1));ax.text(.115,.5,"$x$",transform=ax.transAxes,ha="center",va="center",fontsize=12);ax.add_patch(Rectangle((.29,.4),.10,.20,transform=ax.transAxes,fill=False,ec=TEAL,lw=1));ax.text(.34,.5,"$z$",transform=ax.transAxes,ha="center",va="center",fontsize=12)
    for k,x in enumerate(np.linspace(.48,.70,5)):
        ax.add_patch(Rectangle((x,.43),.055,.14,transform=ax.transAxes,fill=False,ec=VIOLET,lw=.9))
        if k<4:ax.add_artist(FancyArrowPatch((x+.055,.5),(x+.075,.5),transform=ax.transAxes,arrowstyle="->",mutation_scale=8,lw=.7,color=GRAPHITE))
    ax.add_patch(Rectangle((.82,.36),.13,.28,transform=ax.transAxes,fill=False,ec=ORANGE,lw=1));ax.text(.885,.5,r"$\hat{x}$",transform=ax.transAxes,ha="center",va="center",fontsize=12)
    for a,b in [(.18,.29),(.39,.48),(.755,.82)]:ax.add_artist(FancyArrowPatch((a,.5),(b,.5),transform=ax.transAxes,arrowstyle="->",mutation_scale=10,lw=.9,color=GRAPHITE))
    ax.text(.60,.8,"conditioning $c$",transform=ax.transAxes,ha="center")
    for x in np.linspace(.50,.72,4):ax.add_artist(FancyArrowPatch((.60,.76),(x+.025,.59),transform=ax.transAxes,arrowstyle="->",mutation_scale=8,lw=.6,color=STEEL))
    save(fig,11)

def figure12():
    fig,ax=plt.subplots(figsize=(10,4.5));ax.set_axis_off()
    for i in range(6):ax.add_patch(Rectangle((.05+i*.024,.66),.018,.14,transform=ax.transAxes,fill=False,ec=NAVY,lw=.7))
    levels=[(.32,.64,.08,.18),(.43,.57,.08,.32),(.54,.50,.08,.46),(.65,.57,.08,.32),(.76,.64,.08,.18)]
    for x,y,w,h in levels:ax.add_patch(Rectangle((x,y),w,h,transform=ax.transAxes,fill=False,ec=VIOLET,lw=1))
    for a,b in zip(levels[:-1],levels[1:]):x1,y1,w1,h1=a;x2,y2,w2,h2=b;ax.add_artist(FancyArrowPatch((x1+w1,y1+h1/2),(x2,y2+h2/2),transform=ax.transAxes,arrowstyle="->",mutation_scale=9,lw=.8,color=GRAPHITE))
    ax.add_patch(Rectangle((.19,.37),.08,.24,transform=ax.transAxes,fill=False,ec=TEAL,lw=1));ax.text(.23,.49,"$z_t$",transform=ax.transAxes,ha="center")
    for x in [.37,.48,.59,.70]:ax.add_artist(FancyArrowPatch((.12,.66),(x,.78),transform=ax.transAxes,arrowstyle="->",mutation_scale=7,lw=.5,color=STEEL,alpha=.8))
    ax.add_patch(Rectangle((.87,.37),.08,.24,transform=ax.transAxes,fill=False,ec=ORANGE,lw=1));ax.text(.91,.49,r"$\hat{x}$",transform=ax.transAxes,ha="center");ax.add_artist(FancyArrowPatch((.84,.73),(.87,.49),transform=ax.transAxes,arrowstyle="->",mutation_scale=9,lw=.8,color=GRAPHITE));save(fig,12)

def figure13():
    rng=np.random.default_rng(13);fig,axs=plt.subplots(1,3,figsize=(9.1,3.2));Q=rng.normal(size=(8,5));K=rng.normal(size=(6,5));V=rng.normal(size=(6,5));A=Q@K.T/np.sqrt(5);A=np.exp(A-A.max(1,keepdims=True));A/=A.sum(1,keepdims=True)
    for ax,M,xl,yl in [(axs[0],Q@K.T/np.sqrt(5),"$k_j$","$q_i$"),(axs[1],A,"text token","latent position"),(axs[2],A@V,"feature","latent position")]:
        ax.pcolormesh(M,cmap="Blues",shading="nearest");ax.set(xlabel=xl,ylabel=yl);clean(ax)
    axs[0].text(.5,1.05,r"$QK^\top/\sqrt{d}$",transform=axs[0].transAxes,ha="center");axs[1].text(.5,1.05,r"$\mathrm{softmax}(\cdot)$",transform=axs[1].transAxes,ha="center");axs[2].text(.5,1.05,r"$AV$",transform=axs[2].transAxes,ha="center");save(fig,13)

def figure14():
    fig,ax=plt.subplots(figsize=(9.7,4.6));ax.set_axis_off();xs=np.linspace(.12,.78,5)
    for i,x in enumerate(xs):
        ax.add_patch(Rectangle((x,.62),.09,.16,transform=ax.transAxes,fill=False,ec=NAVY,lw=1));ax.add_patch(Rectangle((x,.24),.09,.16,transform=ax.transAxes,fill=False,ec=TEAL,lw=1));ax.add_artist(FancyArrowPatch((x+.045,.40),(x+.045,.62),transform=ax.transAxes,arrowstyle="->",mutation_scale=8,lw=.75,color=VIOLET))
        if i<4:
            ax.add_artist(FancyArrowPatch((x+.09,.70),(xs[i+1],.70),transform=ax.transAxes,arrowstyle="->",mutation_scale=9,lw=.8,color=GRAPHITE));ax.add_artist(FancyArrowPatch((x+.09,.32),(xs[i+1],.32),transform=ax.transAxes,arrowstyle="->",mutation_scale=9,lw=.8,color=GRAPHITE))
    ax.add_patch(Rectangle((.02,.24),.07,.16,transform=ax.transAxes,fill=False,ec=ORANGE,lw=1));ax.text(.055,.32,"$c_s$",transform=ax.transAxes,ha="center",va="center");ax.add_artist(FancyArrowPatch((.09,.32),(.12,.32),transform=ax.transAxes,arrowstyle="->",mutation_scale=9,lw=.8,color=GRAPHITE))
    ax.add_patch(Rectangle((.02,.62),.07,.16,transform=ax.transAxes,fill=False,ec=STEEL,lw=1));ax.text(.055,.70,"$z_t$",transform=ax.transAxes,ha="center",va="center");ax.add_artist(FancyArrowPatch((.09,.70),(.12,.70),transform=ax.transAxes,arrowstyle="->",mutation_scale=9,lw=.8,color=GRAPHITE))
    ax.add_patch(Rectangle((.88,.55),.08,.30,transform=ax.transAxes,fill=False,ec=ORANGE,lw=1));ax.text(.92,.70,r"$\epsilon_\theta$",transform=ax.transAxes,ha="center",va="center");ax.add_artist(FancyArrowPatch((.87,.70),(.88,.70),transform=ax.transAxes,arrowstyle="->",mutation_scale=9,lw=.8,color=GRAPHITE));save(fig,14)

if __name__=="__main__":
    for f in [figure01,figure02,figure03,figure04,figure05,figure06,figure07,figure08,figure09,figure10,figure11,figure12,figure13,figure14]: f()
