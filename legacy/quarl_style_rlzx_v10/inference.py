"""Reuse the actor logits already evaluated for NumPy action sampling."""
import ast
import functools
import inspect
import textwrap

import numpy as np
import torch
from reconstructed_ppo import batch


@functools.lru_cache(maxsize=4)
def evaluator(original):
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    function=tree.body[0]
    assignments=[node for node in function.body if isinstance(node,ast.Assign)
                 and any(isinstance(t,ast.Name)and t.id=='logits'for t in node.targets)]
    assert len(assignments)==1
    assert ast.dump(assignments[0].value)==ast.dump(ast.parse('self.actor(policy_obs)',mode='eval').body)
    assignments[0].value=ast.Name(id='cached_logits',ctx=ast.Load())
    function.args.kwonlyargs.append(ast.arg(arg='cached_logits'))
    function.args.kw_defaults.append(None)
    ast.fix_missing_locations(tree)
    namespace=dict(original.__globals__)
    exec(compile(tree,__file__,'exec'),namespace)
    return namespace[function.name]


def take_training_action(net,obs,rng):
    assert not net.training, 'Inference reuse requires eval mode'
    p=obs[0];ids=p.y[p.y!=-1]
    with torch.no_grad():
        logits=net.actor(p)
        probs=torch.softmax(logits[p.y!=-1].flatten().double(),0).numpy()
        assert len(ids) and np.isfinite(probs).all()
        slot=min(int(np.searchsorted(np.cumsum(probs),rng.random(),side='right')),len(ids)-1)
        _,lp,_,values,_,_=evaluator(net.get_action_and_value.__func__)(
            net,batch([obs]),action=torch.tensor([slot]),device='cpu',cached_logits=logits)
    return slot,int(ids[slot]),float(lp[0]),float(values[0])
