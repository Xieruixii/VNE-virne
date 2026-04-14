import networkx as nx

from virne.core import Controller, Recorder, Counter, Solution, Logger
from virne.network import PhysicalNetwork, VirtualNetwork
from virne.solver.base_solver import SolverRegistry
from virne.solver.heuristic.node_rank import BaseNodeRankSolver
from virne.solver.rank.node_rank import GRCNodeRank


@SolverRegistry.register(solver_name='rdma_always_offload', solver_type='node_ranking')
class RDMAAlwaysOffloadSolver(BaseNodeRankSolver):
    """
    Ablation-3: Always-Offload-If-Eligible (Phase-1 skeleton)

    For each virtual link, if both endpoint physical nodes are RDMA-capable (and optionally have budget),
    mark the link as offloaded (z=1). Otherwise z=0.

    NOTE: Phase-1 does NOT change routing algorithm; it only annotates decisions so that
    later we can add budget constraints, RDMA-path selection, EHPC metrics, etc.
    """

    def __init__(self, controller: Controller, recorder: Recorder, counter: Counter, logger: Logger, config, **kwargs):
        super().__init__(controller, recorder, counter, logger, config, **kwargs)
        # use the same node rank as grc_rank by default
        self.sigma = kwargs.get('sigma', 0.00001)
        self.d = kwargs.get('d', 0.85)
        self.node_rank = GRCNodeRank(sigma=self.sigma, d=self.d)
        self.link_rank = None

        # whether to check budget in eligibility gate (optional for Phase-1)
        self.check_budget = bool(kwargs.get("rdma_check_budget", False))

    def _eligible(self, p_net: PhysicalNetwork, p_u: int, p_v: int) -> bool:
        if p_net.nodes[p_u].get("rdma_capable", 0) != 1:
            return False
        if p_net.nodes[p_v].get("rdma_capable", 0) != 1:
            return False
        if self.check_budget:
            if p_net.nodes[p_u].get("rdma_budget_free", 0) <= 0:
                return False
            if p_net.nodes[p_v].get("rdma_budget_free", 0) <= 0:
                return False
        return True

    def link_mapping(self, v_net: VirtualNetwork, p_net: PhysicalNetwork, solution: Solution) -> bool:
        # Determine z_l for each virtual link, based on endpoint eligibility
        # Save decision into solution (and also into v_net edge attrs for later recorder use)
        rdma_decisions = {}

        for (u, v) in v_net.links:
            p_u = solution.node_slots.get(u, None)
            p_v = solution.node_slots.get(v, None)
            if p_u is None or p_v is None:
                # node mapping incomplete
                rdma_decisions[(u, v)] = 0
                continue

            z = 1 if self._eligible(p_net, p_u, p_v) else 0
            rdma_decisions[(u, v)] = z
            # annotate on v_net edge
            try:
                v_net.links[(u, v)]["z"] = z
            except Exception:
                pass

        solution["rdma_z"] = rdma_decisions

        # Then do normal link mapping (TCP-like) using existing mapper
        if self.link_rank is None:
            sorted_v_links = v_net.links
        else:
            v_net_edges_rank_dict = self.link_rank(v_net)
            v_net_edges_sort = sorted(v_net_edges_rank_dict.items(), reverse=True, key=lambda x: x[1])
            sorted_v_links = [edge_value[0] for edge_value in v_net_edges_sort]

        link_mapping_result = self.controller.link_mapper.link_mapping(
            v_net, p_net,
            solution=solution,
            sorted_v_links=sorted_v_links,
            shortest_method=self.shortest_method,
            k=self.k_shortest,
            inplace=True
        )
        return link_mapping_result