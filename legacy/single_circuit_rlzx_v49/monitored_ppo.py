"""Original full-action PPO update plus read-only convergence instrumentation."""
from runtime import torch,np,time,gae,batch

def ppo_update(net, optimizer, rollout, bootstrap, config, deadline):
    observations, actions, old_logs, values, rewards, dones = zip(*rollout)
    advantages, returns = gae(rewards, dones, values, bootstrap)
    acts = torch.tensor(actions)
    old_logs = torch.tensor(old_logs)
    losses = []; diagnostics = []
    before = torch.cat([p.detach().flatten() for p in net.parameters()]).clone()
    actor_before = torch.cat([p.detach().flatten() for p in net.actor_gnn.parameters()]).clone()
    for _ in range(config['epochs']):
        order = torch.randperm(len(rollout))
        for start in range(0, len(rollout), config['minibatch']):
            if time.perf_counter() >= deadline:
                break
            ix = order[start:start + config['minibatch']]
            _, logs, entropy, v, _, _ = net.get_action_and_value(
                batch([observations[i] for i in ix.tolist()]), action=acts[ix], device='cpu')
            ratio = (logs - old_logs[ix]).exp()
            policy_loss = torch.maximum(-advantages[ix] * ratio,
                                        -advantages[ix] * ratio.clamp(.8, 1.2)).mean()
            loss = policy_loss + .5 * (v - returns[ix]).square().mean() - .01 * entropy.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite PPO loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(net.parameters(), .5)
            if not torch.isfinite(grad):
                raise FloatingPointError('nonfinite PPO gradient')
            optimizer.step()
            losses.append(float(loss.detach()))
            with torch.no_grad():
                log_ratio=logs-old_logs[ix]
                diagnostics.append(dict(policy_loss=float(policy_loss),value_loss=float((v-returns[ix]).square().mean()),entropy=float(entropy.mean()),approx_kl=float(((ratio-1)-log_ratio).mean()),clip_fraction=float(((ratio-1).abs()>.2).float().mean()),gradient_norm=float(grad)))
        if time.perf_counter() >= deadline:
            break
    after = torch.cat([p.detach().flatten() for p in net.parameters()])
    actor_after = torch.cat([p.detach().flatten() for p in net.actor_gnn.parameters()])
    result = dict(samples=len(rollout), optimizer_steps=len(losses),
                mean_loss=float(np.mean(losses)) if losses else None,
                parameter_l2_delta=float(torch.linalg.vector_norm(after - before)),
                actor_parameter_l2_delta=float(torch.linalg.vector_norm(actor_after - actor_before)))
    rng_before=torch.get_rng_state().clone()
    with torch.no_grad():
        packed=batch(list(observations))
        _,logs,entropy,pred,_,_=net.get_action_and_value(packed,action=acts,device='cpu')
        target_var=float(returns.var(unbiased=False))
        ev=lambda v:None if target_var<1e-12 else 1-float((returns-v).var(unbiased=False))/target_var
        policy=packed[0];positions=(policy.y!=-1).nonzero(as_tuple=True)[0]
        groups=policy.batch[positions];legal=policy.y[positions];logits=net.actor(policy).flatten()[positions]
        stop_probs=[];normalized_entropy=[]
        for i in range(len(rollout)):
            mask=groups==i;prob=logits[mask].softmax(0)
            stop_probs.append(float(prob[legal[mask]==9003001].sum()))
            k=int((prob>0).sum());normalized_entropy.append(float(entropy[i])/np.log(k)if k>1 else 0.)
        result['metrics']={k:float(np.mean([d[k]for d in diagnostics]))for k in diagnostics[0]}if diagnostics else{}
        result['metrics'].update(value_explained_variance_before=ev(torch.tensor(values)),value_explained_variance_after=ev(pred),return_mean=float(returns.mean()),return_std=target_var**.5,reward_mean=float(np.mean(rewards)),nonzero_reward_fraction=float(np.mean(np.asarray(rewards)!=0)),positive_reward_fraction=float(np.mean(np.asarray(rewards)>0)),post_entropy=float(entropy.mean()),normalized_entropy=float(np.mean(normalized_entropy)),stop_probability=float(np.mean(stop_probs)),post_approx_kl=float(((logs-old_logs).exp()-1-(logs-old_logs)).mean()))
    assert torch.equal(rng_before,torch.get_rng_state()),'Metrics must not consume RNG'
    return result
