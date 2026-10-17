"""Condition action ranking on the same finite-trajectory context as the critic."""
import torch
from torch import nn
from flat_contextual import FlatContextualAgent

class ActorContextAgent(FlatContextualAgent):
    def __init__(self,parent,width=32,actor_context='full'):
        nn.Module.__init__(self)
        assert actor_context in ['zero','full'];self.actor_context=actor_context
        self.base=parent.base;self.context_value=parent.context_value;self.stop_context=parent.stop_context
        name,projection=list(self.base.actor_gnn.named_children())[-1]
        assert isinstance(projection,nn.Linear)and projection.in_features==width and projection.out_features==1
        self.actor_projection=projection;setattr(self.base.actor_gnn,name,nn.Identity())
        self.actor_residual=nn.Sequential(nn.Linear(width+4,width),nn.ReLU(),nn.Linear(width,1))
        nn.init.zeros_(self.actor_residual[-1].weight);nn.init.zeros_(self.actor_residual[-1].bias)
    @property
    def actor_gnn(self):
        return nn.ModuleList([self.base.actor_gnn,self.actor_projection,self.actor_residual,self.stop_context])
    def actor(self,p):
        features=self.base.actor(p)
        group=p.batch if hasattr(p,'batch')and p.batch is not None else torch.zeros(len(p.y),dtype=torch.long,device=p.y.device)
        ctx=p.context if self.actor_context=='full'else torch.zeros_like(p.context)
        logits=self.actor_projection(features)+self.actor_residual(torch.cat([features,ctx[group]],dim=-1))
        stop=(p.y==9003001).to(logits.dtype).unsqueeze(-1)
        return logits+stop*self.stop_context(p.context)[group]
