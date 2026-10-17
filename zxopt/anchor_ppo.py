"""PPO for the sampled joint anchor and conditional local action.

The entropy bonus is H(anchor)+H(actions at the sampled anchor), a sampled
conditional regularizer. It does not enumerate all anchors to compute the
exact marginal hierarchical entropy or differentiate through discrete sampling.
"""
import time

import numpy as np
import torch

from .anchor_policy import batch
from .legacy import LEGACY
from runtime import gae


def ppo_update(net, optimizer, rollout, bootstrap, config, deadline):
    observations, actions, old_logs, values, rewards, dones = zip(*rollout)
    gamma, lam = config.get("discount_factor", .99), config.get("gae_lambda", .95)
    advantages, returns = gae(rewards, dones, values, bootstrap, gamma=gamma, lam=lam)
    acts, old_logs = torch.tensor(actions), torch.tensor(old_logs)
    before = torch.cat([p.detach().flatten() for p in net.parameters()]).clone()
    actor_before = torch.cat([p.detach().flatten() for p in net.actor_gnn.parameters()]).clone()
    losses, diagnostics = [], []
    for _ in range(config["epochs"]):
        order = torch.randperm(len(rollout))
        for start in range(0, len(rollout), config["minibatch"]):
            if time.perf_counter() >= deadline:
                break
            ix = order[start:start+config["minibatch"]]
            _, logs, entropy, predicted, _, _ = net.get_action_and_value(
                batch([observations[i] for i in ix.tolist()]), action=acts[ix])
            log_ratio = logs-old_logs[ix]; ratio = log_ratio.exp()
            policy_loss = torch.maximum(-advantages[ix]*ratio, -advantages[ix]*ratio.clamp(.8, 1.2)).mean()
            value_loss = (predicted-returns[ix]).square().mean()
            loss = policy_loss+.5*value_loss-.01*entropy.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite anchor PPO loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(net.parameters(), .5)
            if not torch.isfinite(grad):
                raise FloatingPointError("Nonfinite anchor PPO gradient")
            optimizer.step(); losses.append(float(loss.detach()))
            diagnostics.append(dict(policy_loss=float(policy_loss.detach()), value_loss=float(value_loss.detach()),
                                    entropy=float(entropy.detach().mean()),
                                    approx_kl=float((torch.expm1(log_ratio.detach().double())
                                                     -log_ratio.detach().double()).mean()),
                                    clip_fraction=float(((ratio-1).abs()>.2).float().mean()), gradient_norm=float(grad)))
        if time.perf_counter() >= deadline:
            break
    after = torch.cat([p.detach().flatten() for p in net.parameters()])
    actor_after = torch.cat([p.detach().flatten() for p in net.actor_gnn.parameters()])
    state = torch.get_rng_state().clone()
    with torch.no_grad():
        a, p, v = packed = batch(observations)
        la, ha = net.distribution(a, a.selected)
        lm, hm = net.distribution(p, acts)
        logs, entropy, predicted = la+lm, ha+hm, net.get_value(v)
        variance = float(returns.var(unbiased=False))
        explained = lambda values: None if variance < 1e-12 else 1-float((returns-values).var(unbiased=False))/variance
        metrics = {k: float(np.mean([d[k] for d in diagnostics])) for k in diagnostics[0]} if diagnostics else {}
        normalized = []; normalized_anchor = []; normalized_local = []
        for i in range(len(observations)):
            ka = int(((a.y != -1) & (a.batch == i)).sum())
            km = int(((p.y != -1) & (p.batch == i)).sum())
            normalized.append(float(entropy[i])/max(1e-12, np.log(ka)+np.log(km)))
            normalized_anchor.append(float(ha[i])/np.log(ka) if ka > 1 else 0.)
            normalized_local.append(float(hm[i])/np.log(km) if km > 1 else 0.)
        metrics.update(discount_factor=gamma, gae_lambda=lam,
                       anchor_entropy=float(ha.mean()), conditional_action_entropy=float(hm.mean()),
                       post_entropy=float(entropy.mean()), normalized_entropy=float(np.mean(normalized)),
                       normalized_anchor_entropy=float(np.mean(normalized_anchor)),
                       normalized_local_entropy=float(np.mean(normalized_local)),
                       value_explained_variance_before=explained(torch.tensor(values)),
                       value_explained_variance_after=explained(predicted), return_mean=float(returns.mean()),
                       return_std=variance**.5, reward_mean=float(np.mean(rewards)),
                       nonzero_reward_fraction=float(np.mean(np.asarray(rewards) != 0)),
                       positive_reward_fraction=float(np.mean(np.asarray(rewards) > 0)),
                       post_approx_kl=float((torch.expm1((logs-old_logs).double())
                                             -(logs-old_logs).double()).mean()))
    assert torch.equal(state, torch.get_rng_state()), "Diagnostics must not consume policy RNG"
    return dict(samples=len(rollout), optimizer_steps=len(losses),
                mean_loss=float(np.mean(losses)) if losses else None,
                parameter_l2_delta=float(torch.linalg.vector_norm(after-before)),
                actor_parameter_l2_delta=float(torch.linalg.vector_norm(actor_after-actor_before)), metrics=metrics)
