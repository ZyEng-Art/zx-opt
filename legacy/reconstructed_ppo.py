"""Small reconstructed PPO on downloaded original environment; not paper reproduction.

Original default twoqubits reward, full original legal actions and STOP are preserved.
CPU device and recording get_data inputs are the only environment overrides.
"""
import contextlib, hashlib, importlib.util, json, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data, Batch

ROOT=Path('/workspace'); STOP=1792108800


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def source_classes():
    base=ROOT/'Circopt-RL-ZXCalc-main/rl-zx'
    env=load(base/'gym-zx/gym_zx/envs/zx_env.py','ppo_original_env')
    agent=load(base/'rl_agent.py','ppo_original_agent')
    class RecordedEnv(env.ZXEnv):
        def get_data(self,circuit):
            self.recorded_cost_circuit=circuit.copy()
            return super().get_data(circuit)
    return RecordedEnv,agent.AgentGNN


def data(obs):
    p,v=obs
    return (Data(x=p[0],edge_index=p[1],edge_attr=p[2],y=p[3]),
            Data(x=v[0],edge_index=v[1],edge_attr=v[2]))


def batch(observations):
    return tuple(Batch.from_data_list([obs[i] for obs in observations]) for i in range(2))


def seed_reset(env,seed):
    # Explicit generator seed independent of sampled policy episode length.
    random.seed(seed);np.random.seed(seed)
    _,info=env.reset()
    return data(info['graph_obs'])


def main():
    out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
    smoke=len(sys.argv)>2 and sys.argv[2]=='smoke'
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    Env,Agent=source_classes();total=256 if smoke else 16384
    seeds=[11] if smoke else [11,29]
    protocol=dict(scope=__doc__,environment_overrides='env.device=cpu; get_data records a copy without changing its returned stats; suppress original prints',
                  original_reward='Source default gate_type=twoqubits, max_compression10, nonterminal delta plus terminal baseline-relative reward; STOP/full action set retained',
                  seeds=seeds,train_qubits=5,train_depth=70,steps_per_seed=total,rollout=128,minibatch=32,epochs=4,
                  gamma=.99,gae_lambda=.95,clip=.2,lr=.0003,entropy=.01,value_weight=.5,max_grad_norm=.5,
                  width=32,device='cpu',reset_seed_start=26050000,checkpoint_selection='Final fixed step checkpoint only, no validation/test selection',
                  sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'Circopt-RL-ZXCalc-main/rl-zx/rl_agent.py',ROOT/'Circopt-RL-ZXCalc-main/rl-zx/gym-zx/gym_zx/envs/zx_env.py']},
                  limitations='Trainer reconstructed because downloaded source lacks training loop/checkpoint. Single-environment short CPU pilot, not16h paper training or exact original hyperparameters/software. Up-to-permutation extraction preserved; native reward is not independently verified fixed-wire quality. No Quarl-trained comparison.')
    protocol['resume_artifacts']='Separate trusted local resume snapshots every4096steps: optimizer, RNGs, env state/current observation and partial episode. Model-only checkpoints remain weights_only-loadable. Resume serialization failures logged separately; source training behavior unchanged.'
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));reports=[]
    for seed in seeds:
        torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
        net=Agent(None,'cpu',c_hidden=32,c_hidden_v=32)
        torch.save(dict(state_dict=net.state_dict(),seed=seed,width=32,steps=0),out/f'initial_{seed}.pt')
        optimizer=torch.optim.Adam(net.parameters(),lr=.0003,eps=1e-5)
        env=Env(5,70);env.device='cpu';episode=0;ep_reward=0.;ep_steps=0;records=[];timers=dict(reset=0.,step=0.,forward=0.,update=0.)
        started=time.perf_counter();null=open(os.devnull,'w')
        with contextlib.redirect_stdout(null):
            t=time.perf_counter();obs=seed_reset(env,26050000+episode);timers['reset']+=time.perf_counter()-t
        for offset in range(0,total,128):
            if time.time()>STOP-900:
                torch.save(dict(state_dict=net.state_dict(),seed=seed,width=32,steps=offset,status='deadline_partial'),out/f'partial_{seed}.pt')
                raise RuntimeError('Deadline guard: partial training preserved, not counted as final model')
            observations=[];actions=[];logprobs=[];values=[];rewards=[];dones=[]
            for step in range(128):
                observations.append(obs);t=time.perf_counter()
                with torch.no_grad():a,lp,entropy,value,_,action_id=net.get_action_and_value(batch([obs]),device='cpu')
                timers['forward']+=time.perf_counter()-t
                actions.append(int(a.item()));logprobs.append(float(lp.item()));values.append(float(value.item()))
                t=time.perf_counter()
                with contextlib.redirect_stdout(null):_,reward,done,truncated,info=env.step(int(action_id.item()))
                timers['step']+=time.perf_counter()-t
                assert not truncated and np.isfinite(reward)
                rewards.append(float(reward));dones.append(bool(done));ep_reward+=float(reward);ep_steps+=1
                if done:
                    rec=dict(seed=seed,episode=episode,reset_seed=26050000+episode,train_step=offset+step+1,reward=ep_reward,length=ep_steps,
                             native_final_twoqubits=info['rl_stats']['twoqubits'],native_final_gflow_twoqubits=info['rl_gates'],native_initial_twoqubits=env.initial_stats[env.gate_type],native_baseline_twoqubits=min(env.pyzx_gates,env.korb_gates,env.initial_stats[env.gate_type]),
                             final_action=int(action_id.item()),final_action_type=info['action'],input_sha256=hashlib.sha256(env.rand_circuit.to_qasm().encode()).hexdigest())
                    rec['dense_reward_sum']=(rec['native_initial_twoqubits']-rec['native_final_gflow_twoqubits'])/env.max_compression
                    rec['terminal_reward']=(rec['native_baseline_twoqubits']-rec['native_final_twoqubits'])/env.max_compression
                    assert np.isclose(ep_reward,rec['dense_reward_sum']+rec['terminal_reward'],atol=1e-7)
                    records.append(rec)
                    with (out/'episodes.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
                    episode+=1;ep_reward=0.;ep_steps=0
                    with contextlib.redirect_stdout(null):
                        t=time.perf_counter();obs=seed_reset(env,26050000+episode);timers['reset']+=time.perf_counter()-t
                else:obs=data(info['graph_obs'])
            with torch.no_grad():bootstrap=float(net.get_value(batch([obs])[1]).item())
            adv=torch.zeros(128);gae=0.
            for i in reversed(range(128)):
                next_value=bootstrap if i==127 else values[i+1];mask=1.-float(dones[i])
                delta=rewards[i]+.99*next_value*mask-values[i]
                gae=delta+.99*.95*mask*gae;adv[i]=gae
            old_value=torch.tensor(values);returns=adv+old_value;adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
            old_log=torch.tensor(logprobs);act=torch.tensor(actions);losses=[];kls=[];t=time.perf_counter()
            for epoch in range(4):
                order=torch.randperm(128)
                for start in range(0,128,32):
                    ix=order[start:start+32];_,lp,entropy,value,_,_=net.get_action_and_value(batch([observations[i] for i in ix.tolist()]),action=act[ix],device='cpu')
                    log_ratio=lp-old_log[ix];ratio=log_ratio.exp()
                    policy=torch.maximum(-adv[ix]*ratio,-adv[ix]*ratio.clamp(.8,1.2)).mean()
                    loss=policy+.5*(value-returns[ix]).square().mean()-.01*entropy.mean()
                    assert torch.isfinite(loss).item()
                    optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),.5);optimizer.step()
                    losses.append(float(loss.detach()));kls.append(float(((ratio-1)-log_ratio).mean().detach()))
            timers['update']+=time.perf_counter()-t
            progress=dict(seed=seed,steps=offset+128,episodes=episode,loss=float(np.mean(losses)),approx_kl=float(np.mean(kls)),seconds=time.perf_counter()-started,timers=timers.copy())
            with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(progress)+'\n')
            if (offset+128)%1024==0 or smoke:print(json.dumps(progress),flush=True)
            if (offset+128)%4096==0 or smoke:
                torch.save(dict(state_dict=net.state_dict(),seed=seed,width=32,steps=offset+128),out/f'checkpoint_{seed}_{offset+128}.pt')
                try:
                    snapshot=dict(state_dict=net.state_dict(),optimizer_state_dict=optimizer.state_dict(),seed=seed,width=32,steps=offset+128,episode_index=episode,partial_episode_reward=ep_reward,partial_episode_steps=ep_steps,env_state=env.__dict__,current_observation=obs,torch_rng_state=torch.get_rng_state(),python_rng_state=random.getstate(),numpy_rng_state=np.random.get_state(),timers=timers.copy(),first32episodes=records[:32],last32episodes=records[-32:],source_sha256=protocol['sources'],trust='Own generated research artifact; contains Python objects, load only as trusted private resume state')
                    temporary=out/f'resume_{seed}.tmp';torch.save(snapshot,temporary);os.replace(temporary,out/f'resume_{seed}_{offset+128}.pt')
                except Exception as exc:
                    with (out/'resume_errors.jsonl').open('a') as f:f.write(json.dumps(dict(seed=seed,steps=offset+128,error_type=type(exc).__name__,error=str(exc)))+'\n')
        torch.save(dict(state_dict=net.state_dict(),seed=seed,width=32,steps=total),out/f'final_{seed}.pt')
        null.close();reports.append(dict(seed=seed,steps=total,episodes=episode,seconds=time.perf_counter()-started,timers=timers,mean_first32_reward=float(np.mean([r['reward'] for r in records[:32]])),mean_last32_reward=float(np.mean([r['reward'] for r in records[-32:]]))))
        (out/'training_report.json').write_text(json.dumps(dict(status='completed' if len(reports)==len(seeds) else 'partial',seeds=reports),indent=2))


if __name__=='__main__':main()
