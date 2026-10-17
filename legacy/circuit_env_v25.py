"""Initialize the pinned rewrite environment from a supplied unitary NAM circuit."""
from pathlib import Path
import pyzx as zx
from runtime import source_classes as original_classes
from cflow_transfer_probe import clean

def source_classes(source_dir,input_path):
    Parent,Agent=original_classes(source_dir)
    text=Path(input_path).read_text()
    class CircuitEnv(Parent):
        def reset(self,*args,**kwargs):
            self._matching_graph=self._cached_lcomp=self._cached_ids=None
            self.episode_len=0;self.episode_reward=0.;self.action_pattern=[];self.max_reward=0.
            self.opt_episode_len=0;self.swap_cost=0;self.initial_depth=0
            self.episode_stats=dict.fromkeys(['pivb','pivg','piv','lc','id','gf'],0)
            self.best_action_stats=self.episode_stats.copy()
            circuit=zx.Circuit.from_qasm(text).to_basic_gates()
            assert circuit.qubits==self.qubits
            self.rand_circuit=circuit;self.final_circuit=circuit
            self.no_opt_stats=self.initial_stats=self.get_data(circuit)
            self.min_gates=self.current_gates=self.initial_stats['gates']
            self.graph=clean(circuit).to_graph()
            zx.teleport_reduce(self.graph);self.to_graph_like();self.graph=self.graph.copy()
            assert self.graph.num_vertices()<self.shape,'Graph exceeds upstream action encoding bound'
            self.pivot_info_dict=self.match_pivot_parallel()|self.match_pivot_boundary()|self.match_pivot_gadget()
            self.gadget_info_dict,self.gadgets=self.match_phase_gadgets();self.gadget_fusion_ids=list(self.gadget_info_dict)
            obs=[self.policy_obs(),self.value_obs()];self._last_graph_obs=obs
            return self.graph,dict(graph_obs=obs)
        def step(self,action):
            assert self.graph.num_vertices()<self.shape,'Graph exceeds upstream action encoding bound'
            return super().step(action)
    return CircuitEnv,Agent
