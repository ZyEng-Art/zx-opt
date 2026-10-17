"""Global critic and full-action PPO for globally coupled extraction cost."""
import torch
from torch import nn

class FlatContextualAgent(nn.Module):
    def __init__(self,base,width=32):
        super().__init__();self.base=base
        self.context_value=nn.Sequential(nn.Linear(4,width),nn.ReLU(),nn.Linear(width,1))
        nn.init.zeros_(self.context_value[-1].weight);nn.init.zeros_(self.context_value[-1].bias)
        self.stop_context=nn.Linear(4,1,bias=False);nn.init.zeros_(self.stop_context.weight)
    @property
    def actor_gnn(self):return self.base.actor_gnn
    def actor(self,p):
        logits=self.base.actor(p)
        group=p.batch if hasattr(p,'batch')and p.batch is not None else torch.zeros(len(p.y),dtype=torch.long,device=p.y.device)
        stop=(p.y==9003001).to(logits.dtype).unsqueeze(-1)
        return logits+stop*self.stop_context(p.context)[group]
    def critic(self,v):return self.base.critic(v)+self.context_value(v.context)
    def get_value(self,v):return self.critic(v).flatten()
    def get_action_and_value(self,x,action=None,device='cpu',testing=False):
        policy_obs,value_obs=x
        logits=self.actor(policy_obs)
        values=self.get_value(value_obs)
        positions=(policy_obs.y!=-1).nonzero(as_tuple=True)[0]
        groups=policy_obs.batch[positions];graphs=policy_obs.num_graphs
        counts=torch.bincount(groups,minlength=graphs);assert (counts>0).all()
        selected_logits=logits.flatten()[positions]
        maximum=torch.full((graphs,),-torch.inf,device=selected_logits.device)
        maximum.scatter_reduce_(0,groups,selected_logits.detach(),reduce='amax',include_self=True)
        shifted=selected_logits-maximum[groups]
        mass=torch.zeros_like(maximum).scatter_add_(0,groups,shifted.exp())
        logs=shifted-mass.log()[groups]
        entropy=torch.zeros_like(maximum).scatter_add_(0,groups,-logs.exp()*logs)
        if action is None:
            action=torch.stack([torch.multinomial(logs[groups==g].exp(),1)[0]for g in range(graphs)])
        offsets=counts.cumsum(0)-counts
        flat_indices=offsets+action.to(offsets.device)
        return action,logs[flat_indices],entropy,values,None,None
