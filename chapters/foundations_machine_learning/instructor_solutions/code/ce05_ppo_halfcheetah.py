"""PPO sensitivity experiment on HalfCheetah-v4 using Stable-Baselines3."""
import argparse, json

def main():
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    ap=argparse.ArgumentParser(); ap.add_argument('--steps',type=int,default=200000); ap.add_argument('--out',default='ce05_results.json'); a=ap.parse_args()
    configs=[(3e-4,.2,0.0,.5),(1e-4,.1,0.0,.5),(1e-3,.2,.01,.5)]
    results=[]
    for lr,clip,ent,gclip in configs:
        env=gym.make('HalfCheetah-v4'); model=PPO('MlpPolicy',env,learning_rate=lr,clip_range=clip,ent_coef=ent,max_grad_norm=gclip,seed=7,verbose=0)
        model.learn(a.steps); mean,std=evaluate_policy(model,env,n_eval_episodes=10,deterministic=True)
        results.append({'lr':lr,'clip':clip,'entropy_coef':ent,'grad_clip':gclip,'return_mean':mean,'return_std':std}); env.close()
    open(a.out,'w').write(json.dumps(results,indent=2)); print(json.dumps(results,indent=2))
if __name__=='__main__': main()
