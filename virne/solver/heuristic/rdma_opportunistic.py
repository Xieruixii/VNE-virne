from virne.core import Controller, Recorder, Counter, Solution, Logger
from virne.network import PhysicalNetwork, VirtualNetwork
from virne.solver.base_solver import SolverRegistry

from .rdma_always_offload import RDMAAlwaysOffloadSolver
from .node_rank import BaseNodeRankSolver


@SolverRegistry.register(solver_name='rdma_opportunistic', solver_type='node_ranking')
class RDMAOpportunisticSolver(RDMAAlwaysOffloadSolver):
    def __init__(self, controller: Controller, recorder: Recorder, counter: Counter, logger: Logger, config, **kwargs):
        super().__init__(controller, recorder, counter, logger, config, **kwargs)
        self.utility_threshold = float(kwargs.get("rdma_utility_threshold", 1.0))
        self.ehpc_gain_weight = float(kwargs.get("rdma_ehpc_gain_weight", 0.02))
        self.delay_risk_weight = float(kwargs.get("rdma_delay_risk_weight", 1.0))
        self.fallback_risk_weight = float(kwargs.get("rdma_fallback_risk_weight", 0.5))
        self.traffic_weights = {1: 1.0, 2: 0.6, 3: 0.2}

    def _compute_utility(self, v_net: VirtualNetwork, p_net: PhysicalNetwork, v_link, p_u: int, p_v: int, penalty: float) -> float:
        edge_attr = v_net.links[v_link]
        traffic_type = int(edge_attr.get("traffic_type", 2))
        m = float(edge_attr.get("m", 0))
        delay_sensitive = int(edge_attr.get("d", 0))
        rdma_score_u = float(p_net.nodes[p_u].get("rdma_score", 0.0))
        rdma_score_v = float(p_net.nodes[p_v].get("rdma_score", 0.0))

        ehpc_gain = self.ehpc_gain_weight * m * self.traffic_weights.get(traffic_type, 0.6)
        delay_risk = self.delay_risk_weight * delay_sensitive
        fallback_risk = self.fallback_risk_weight * max(0.0, 1.0 - min(rdma_score_u, rdma_score_v))
        return ehpc_gain - float(penalty) - delay_risk - fallback_risk

    def link_mapping(self, v_net: VirtualNetwork, p_net: PhysicalNetwork, solution: Solution) -> bool:
        rdma_decisions = {}
        rdma_penalties = {}
        rdma_utilities = {}

        for (u, v) in v_net.links:
            p_u = solution.node_slots.get(u, None)
            p_v = solution.node_slots.get(v, None)
            if p_u is None or p_v is None:
                rdma_decisions[(u, v)] = 0
                rdma_penalties[(u, v)] = 0.0
                rdma_utilities[(u, v)] = 0.0
                continue

            eligible, penalty = self._rdma_feasibility_and_penalty(p_net, p_u, p_v)
            utility = self._compute_utility(v_net, p_net, (u, v), p_u, p_v, penalty) if eligible else 0.0
            z = 1 if eligible and utility >= self.utility_threshold else 0
            rdma_decisions[(u, v)] = z
            rdma_penalties[(u, v)] = float(penalty if z else 0.0)
            rdma_utilities[(u, v)] = float(utility)
            try:
                v_net.links[(u, v)]["z"] = z
                v_net.links[(u, v)]["rdma_path_penalty"] = rdma_penalties[(u, v)]
                v_net.links[(u, v)]["rdma_utility"] = rdma_utilities[(u, v)]
            except Exception:
                pass

        solution["rdma_z"] = rdma_decisions
        solution["rdma_path_penalty"] = rdma_penalties
        solution["rdma_utility"] = rdma_utilities
        return BaseNodeRankSolver.link_mapping(self, v_net, p_net, solution)
