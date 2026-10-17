"""One supplied circuit, one shared state pool per run, frozen or online PPO search."""
import contextlib,copy,hashlib,json,os,random,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;R=HERE.parent
sys.path.insert(0,str(R/'quarl_style_rlzx_v10'));sys.path.insert(0,str(HERE))
import numpy as np
import torch
from qiskit import qasm2
from runtime import data,batch,snapshot,restore,nam,direct_cflow,cost,digest,take_action,take_training_action
from buffer import CircuitBuffer,Entry,score
from circuit_env_v25 import source_classes
from completion_feedback_v37 import FeedbackCompletion
from agent import SingleCircuitAgent
from monitored_ppo import ppo_update
def attach(env,obs,current,start,record,root):
    scale=max(1,root);context=torch.tensor([[(current-start)/scale,(record-start)/scale,env.episode_len/env.max_episode_len,start/scale]],dtype=torch.float32)
    for g in obs:g.context=context.clone()
    return obs
def atomic_save(value,path):
    temporary=path.with_suffix('.tmp');torch.save(value,temporary);os.replace(temporary,path)
def worker(conn,spec,method,checkpoint_path,source_dir,config,search_seed):
    try:
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        random.seed(config['model_seed']);np.random.seed(config['model_seed']);torch.manual_seed(config['model_seed'])
        Env,Base=source_classes(source_dir,spec['input_path']);env=Env(spec['qubits'],spec['depth']);env.device='cpu';env.gate_type='gates'
        null=open(os.devnull,'w')
        with contextlib.redirect_stdout(null):_,info=env.reset()
        obs=data(info['graph_obs']);source=qasm2.dumps(nam(env.rand_circuit))+'\n'
        cache=FeedbackCompletion(config['cache_capacity'],'clean');key,root_result,_=cache.evaluate(env.graph)
        obs=attach(env,obs,root_result['cost']['gates'],root_result['cost']['gates'],root_result['cost']['gates'],root_result['cost']['gates'])
        root=Entry(key,root_result['cost'],snapshot(env,obs));archive=CircuitBuffer(root,config['buffer_capacity'])
        candidates={'input':source,'zero_action':root_result['qasm']};errors={}
        try:candidates['direct_cflow']=qasm2.dumps(direct_cflow(env.graph))+'\n'
        except Exception as e:errors['direct_cflow']=repr(e)
        candidates={n:cache.processor.evaluate(dict(qasm=s,sha256=digest(s),cost=cost(qasm2.loads(s))))[0]['qasm']for n,s in candidates.items()}
        baselines={}
        for objective in ['nam','cx']:
            text=min(candidates.values(),key=lambda s:score(cost(qasm2.loads(s)),objective));baselines[objective]=dict(qasm=text,sha256=digest(text),cost=cost(qasm2.loads(text)))
        net=None;optimizer=None;initial_sha=None;root_logs=None
        initial_path=Path(config['adapted_checkpoint']).parent/'initial.pt'
        if method!='uniform':
            c=torch.load(checkpoint_path,map_location='cpu',weights_only=True);assert c['steps']==0 and c['objective']=='nam'
            base=Base(None,'cpu',c_hidden=c['width'],c_hidden_v=c['width']);base.load_state_dict(c['state_dict']);net=SingleCircuitAgent(base,width=c['width']);net.eval()
            state={k:v.detach().cpu()for k,v in net.state_dict().items()}
            h=hashlib.sha256()
            for name,v in sorted(state.items()):h.update(name.encode());h.update(v.numpy().tobytes())
            initial_sha=h.hexdigest();atomic_save(dict(state_dict=state,steps=0,width=c['width'],objective='nam',architecture='single_circuit_masked48',input_sha256=digest(source),scope='Freshstep0base pluszero-lastfullcontextresidual. NonSTOPmaskinsideactor, samePPOdensity.'),initial_path)
            with torch.no_grad():
                ids=root.payload[1][0].y;eligible=(ids!=-1)&(ids!=9003001)
                if not eligible.any():eligible=ids!=-1
                root_logs=net.actor(root.payload[1][0]).flatten()[eligible].double().log_softmax(0)
            if method.startswith('online_'):optimizer=torch.optim.Adam(net.parameters(),lr=config['lr'],eps=1e-5)
        rng=np.random.default_rng(search_seed);torch.manual_seed(218000000+config['model_seed']);random.seed(config['model_seed']);np.random.seed(config['model_seed'])
        conn.send(dict(kind='ready',source=source,input_sha256=digest(source),baselines=baselines,baseline_errors=errors));conn.recv()
        started=time.perf_counter();deadline=started+config['budget_seconds'];active=False;steps=trajectory=updates=0;rollout=[];seen=set();root_best=baselines['nam']['cost']['gates'];last_save=started
        def save():
            if net is None:return
            item=dict(state_dict={k:v.detach().cpu()for k,v in net.state_dict().items()},optimizer_state_dict=None if optimizer is None else optimizer.state_dict(),steps=steps,updates=updates,width=c['width'],objective='nam',architecture='single_circuit_masked48',input_sha256=digest(source),buffer_summary=archive.summary(),method=method,scope='Model/Adamcheckpointonly; notfullenv/archives/RNGresume')
            atomic_save(item,Path(config['adapted_checkpoint']))
        if not bool(((obs[0].y!=-1)&(obs[0].y!=9003001)).any()):
            conn.send(dict(kind='finished',transitions=0,no_legal_rewrites=True));return
        while time.perf_counter()<deadline:
            if not active:
                start=archive.choose('diverse',rng);obs=restore(env,start);previous=start.counts;record=previous['gates'];distance=start.distance;trajectory+=1;key=start.key
                obs=attach(env,obs,previous['gates'],start.counts['gates'],record,root.counts['gates']);active=True
            conn.send(dict(kind='stage',stage='single_circuit_online_search'))
            old_obs=obs;old_key=key;selection=None
            if method=='uniform':
                ids=obs[0].y[obs[0].y!=-1];slots=(ids!=9003001).nonzero(as_tuple=True)[0].tolist()
                if not slots:slots=list(range(len(ids)))
                slot=slots[min(int(rng.random()*len(slots)),len(slots)-1)];action=int(ids[slot])
            elif optimizer is None:slot,action=take_action(net,obs,rng)
            else:
                slot,action,lp,value=take_training_action(net,obs,rng);selection=dict(slot=slot,action=action,joint_log=lp,value=value)
            legal_nonstop=int(((obs[0].y!=-1)&(obs[0].y!=9003001)).sum());assert not legal_nonstop or action!=9003001
            with contextlib.redirect_stdout(null):_,_,done,truncated,info=env.step(action)
            assert not truncated;obs=data(info['graph_obs']);key,candidate,hit=cache.evaluate(env.graph);seen.add(key)
            previous_record=record;record=min(record,candidate['cost']['gates']);root_best=min(root_best,candidate['cost']['gates'])
            r=((previous['gates']-candidate['cost']['gates'])if method=='online_current'else(previous_record-record))/10
            terminal=bool(done or candidate['cost']['gates']>config['alpha']*max(1,root.counts['gates']))
            distance+=int(action!=9003001);obs=attach(env,obs,candidate['cost']['gates'],start.counts['gates'],record,root.counts['gates'])
            admitted=False
            if key not in archive.entries and score(candidate['cost'])<=score(start.counts):admitted=archive.add(Entry(key,candidate['cost'],snapshot(env,obs),distance),start.counts)
            steps+=1
            conn.send(dict(kind='candidate',**candidate,graph_key=key,old_graph_key=old_key,transition=steps,trajectory=trajectory,start_key=start.key,start_distance=start.distance,distance=distance,action=action,reward=r,previous_cost=previous,previous_record=previous_record,record_best=record,terminal=terminal,cache_hit=hit,admitted=admitted,buffer=archive.summary(),phase='rewrite',objective='nam',cost_mode='clean',native_cost=cache.last_native['cost'],clean_cost=cache.last_clean['cost'],context=obs[1].context.tolist(),episode_len=env.episode_len,start_gates=start.counts['gates'],root_gates=root.counts['gates'],reward_mode=('current'if method=='online_current'else'record_best'),legal_nonstop=legal_nonstop,selection=selection,unique_states=len(seen),updates=updates,initial_tensor_sha256=initial_sha))
            if optimizer is not None:rollout.append((old_obs,slot,selection['joint_log'],selection['value'],r,terminal))
            previous=candidate['cost'];active=not terminal
            if optimizer is not None and len(rollout)>=config['rollout']:
                bootstrap=0.
                if active:
                    with torch.no_grad():bootstrap=float(net.get_value(batch([obs])[1])[0])
                t=time.perf_counter();result=ppo_update(net,optimizer,rollout,bootstrap,config,deadline);updates+=1
                result.update(transition=steps,update=updates,seconds=time.perf_counter()-t,elapsed=time.perf_counter()-started,buffer=archive.summary(),best_gates=root_best,reward_mode=method.removeprefix('online_'))
                with torch.no_grad():
                    p=root.payload[1][0];ids=p.y;eligible=(ids!=-1)&(ids!=9003001)
                    if not eligible.any():eligible=ids!=-1
                    lp=net.actor(p).flatten()[eligible].double().log_softmax(0);prob=lp.exp()
                    result['metrics'].update(root_policy_KL_to_initial=float((prob*(lp-root_logs)).sum()),root_policy_TV_to_initial=float((prob-root_logs.exp()).abs().sum()/2))
                conn.send(dict(kind='update',**result));rollout=[]
                if steps%1024==0 or time.perf_counter()-last_save>30:save();last_save=time.perf_counter()
        save();conn.send(dict(kind='finished',transitions=steps,updates=updates,buffer=archive.summary(),best_gates=root_best))
    except(BrokenPipeError,EOFError):pass
    except BaseException:
        try:conn.send(dict(kind='error',traceback=traceback.format_exc()))
        except(BrokenPipeError,EOFError):pass
    finally:conn.close()
