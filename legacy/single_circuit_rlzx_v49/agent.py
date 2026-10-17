"""Full-context action ranking with a consistent nonSTOP policy and PPO density."""
import torch
from actor_contextual import ActorContextAgent
from flat_contextual import FlatContextualAgent
class SingleCircuitAgent(ActorContextAgent):
    def __init__(self,base,width=32):
        super().__init__(FlatContextualAgent(base,width=width),width=width,actor_context='full')
    def actor(self,p):
        logits=super().actor(p)
        group=p.batch if hasattr(p,'batch')and p.batch is not None else torch.zeros(len(p.y),dtype=torch.long,device=p.y.device)
        eligible=(p.y!=-1)&(p.y!=9003001)
        counts=torch.bincount(group[eligible],minlength=int(group.max())+1)
        mask=(p.y==9003001)&(counts[group]>0)
        return logits.masked_fill(mask.unsqueeze(-1),-1e9)
