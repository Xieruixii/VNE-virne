import copy
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, cast

import networkx as nx
from itertools import islice
from virne.solver.base_solver import SolverRegistry
from virne.solver.heuristic.node_rank import BaseNodeRankSolver
from virne.solver.rank.node_rank import NRMNodeRank
from virne.utils import path_to_links


@SolverRegistry.register(solver_name='rdma_rank', solver_type='node_ranking')
class RDMARankSolver(BaseNodeRankSolver):
    """
    RDMA-aware heuristic solver for Virne.

    Expected custom attributes
    --------------------------
    Physical node attrs:
      - cpu
      - rdma_capable : {0,1}
      - nic_slots    : int
      - rack_id      : int
      - qp_budget    : int (optional)

    Physical link attrs:
      - bw
      - latency
      - rdma_link    : {0,1} (optional)

    Virtual link attrs:
      - bw
      - msg_rate
      - delay_sensitive : {0,1}
      - flow_type
      - latency         : optional latency demand
    """

    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        super().__init__(controller, recorder, counter, logger, config, **kwargs)

        self.node_rank = NRMNodeRank()
        self.link_rank = None

        # Solver hyperparameters
        # self.k_shortest = kwargs.get("k_shortest", self.k_shortest)
        # self.shortest_method = kwargs.get("shortest_method", self.shortest_method)
        # Solver hyperparameters
        self.k_shortest = getattr(config.solver, "k_shortest", self.k_shortest)
        self.shortest_method = getattr(config.solver, "shortest_method", self.shortest_method)

        self.alpha_host = getattr(config.solver, "alpha_host", 1.0)
        self.beta_lat = getattr(config.solver, "beta_lat", 1.0)
        self.gamma_nic = getattr(config.solver, "gamma_nic", 1.0)

        self.offload_threshold = getattr(config.solver, "offload_threshold", 0.5)

        self.w_base = getattr(config.solver, "w_base", 1.0)
        self.w_nic = getattr(config.solver, "w_nic", 0.8)
        self.w_neighbor = getattr(config.solver, "w_neighbor", 1.2)
        self.w_rdma_pair = getattr(config.solver, "w_rdma_pair", 1.5)

        # new strategy params
        self.rdma_margin = getattr(config.solver, "rdma_margin", 1.0)
        self.lambda_nic_scarcity = getattr(config.solver, "lambda_nic_scarcity", 1.0)
        self.lambda_slack = getattr(config.solver, "lambda_slack", 0.05)

        self.node_rdma_aware = True
        self.mode_policy = "opportunistic"   # opportunistic | always | disabled


    # ============================================================
    # helpers
    # ============================================================
    def _edge_data(self, g, edge):
        u, v = edge
        if g.has_edge(u, v):
            return g.edges[u, v]
        if g.has_edge(v, u):
            return g.edges[v, u]
        return {}

    def _node_attr(self, g, n, key, default=0):
        return g.nodes[n].get(key, default)

    def _link_attr(self, g, edge, key, default: Any = 0):
        return self._edge_data(g, edge).get(key, default)

    def _is_rdma_node(self, p_net, p_node) -> bool:
        return int(self._node_attr(p_net, p_node, "rdma_capable", 0)) == 1

    def _has_nic_budget(self, p_net, p_node) -> bool:
        return self._node_attr(p_net, p_node, "nic_slots", 0) > 0

    def _same_rack(self, p_net, p1, p2) -> bool:
        return self._node_attr(p_net, p1, "rack_id", -1) == self._node_attr(p_net, p2, "rack_id", -2)

    def _vlink_bw_demand(self, v_net, v_link) -> float:
        demand = 0.0
        for attr in v_net.get_link_attrs(['resource']):
            demand += float(self._link_attr(v_net, v_link, attr.name, 0.0))
        return demand

    def _vlink_latency_demand(self, v_net, v_link) -> Optional[float]:
        value = self._link_attr(v_net, v_link, "latency", None)
        if value is None:
            return None
        return float(value)

    def _path_latency(self, p_net, path: List[int]) -> float:
        return sum(float(self._link_attr(p_net, e, "latency", 1.0)) for e in path_to_links(path))

    def _path_slack(self, p_net, path: List[int], demand_bw: float) -> float:
        """
        Residual bandwidth margin along a path.
        Larger is better.
        """
        return self._path_min_bw(p_net, path) - demand_bw


    def _nic_scarcity_penalty(self, p_net, p_u, p_v) -> float:
        """
        Penalize using RDMA endpoints when NIC slots are scarce.
        """
        nic_u = float(self._node_attr(p_net, p_u, "nic_slots", 0))
        nic_v = float(self._node_attr(p_net, p_v, "nic_slots", 0))
        return self.lambda_nic_scarcity * (
            1.0 / (nic_u + 1e-6) + 1.0 / (nic_v + 1e-6)
        )

    def _path_min_bw(self, p_net, path: List[int]) -> float:
        links = path_to_links(path)
        if not links:
            return float("inf")
        return min(float(self._link_attr(p_net, e, "bw", 0.0)) for e in links)

    def _path_cross_rack_count(self, p_net, path: List[int]) -> int:
        cnt = 0
        for a, b in path_to_links(path):
            if not self._same_rack(p_net, a, b):
                cnt += 1
        return cnt

    def _rdma_subgraph(self, p_net):
        g = nx.DiGraph() if p_net.is_directed() else nx.Graph()

        for n in p_net.nodes:
            if self._is_rdma_node(p_net, n):
                g.add_node(n, **p_net.nodes[n])

        for u, v, data in p_net.edges(data=True):
            if u not in g.nodes or v not in g.nodes:
                continue
            if "rdma_link" in data and int(data.get("rdma_link", 0)) != 1:
                continue
            g.add_edge(u, v, **data)

        return g

    def _feasible_path(
        self,
        g_for_check,
        path: Optional[List[int]],
        demand_bw: float,
        latency_limit: Optional[float] = None,
    ) -> bool:
        if path is None or len(path) == 0:
            return False
        if self._path_min_bw(g_for_check, path) < demand_bw:
            return False
        if latency_limit is not None and self._path_latency(g_for_check, path) > latency_limit:
            return False
        return True


    def _k_shortest_paths(self, g, src, dst, k=10, weight="latency") -> List[List[int]]:
        if src == dst:
            return [[src]]
        try:
            gen = nx.shortest_simple_paths(g, src, dst, weight=weight)
            return list(islice(gen, k))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def _choose_fast_path(
        self,
        p_net,
        src,
        dst,
        demand_bw: float,
        latency_limit: Optional[float] = None,
        rdma_only: bool = False,
    ) -> Optional[List[int]]:
        """
        Fast path selection for node_mapping():
        only find one shortest feasible path, no k-shortest enumeration.
        """
        g = self._rdma_subgraph(p_net) if rdma_only else p_net

        if src == dst:
            return [src]

        def has_enough_bw(u, v):
            return float(self._link_attr(g, (u, v), "bw", 0.0)) >= demand_bw

        available_g = nx.subgraph_view(g, filter_edge=has_enough_bw)

        try:
            path = nx.shortest_path(available_g, src, dst, weight="latency")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

        if latency_limit is not None and self._path_latency(g, path) > latency_limit:
            return None

        return path

    def _choose_best_path(
        self,
        p_net,
        src,
        dst,
        demand_bw: float,
        latency_limit: Optional[float] = None,
        rdma_only: bool = False,
    ) -> Optional[List[int]]:
        """
        Select the best path from the first k shortest feasible candidates.

        Steps:
        1) Build candidate graph (RDMA-only or full graph)
        2) Filter out edges without enough residual bandwidth
        3) Enumerate first k shortest simple paths by latency
        4) Keep feasible paths under latency limit
        5) Choose the one with minimum composite path score
        """
        g = self._rdma_subgraph(p_net) if rdma_only else p_net

        if src == dst:
            return [src]

        def has_enough_bw(u, v):
            return float(self._link_attr(g, (u, v), "bw", 0.0)) >= demand_bw

        available_g = nx.subgraph_view(g, filter_edge=has_enough_bw)

        # 枚举前 k 条最短候选路径
        candidate_paths = self._k_shortest_paths(
            available_g,
            src,
            dst,
            k=self.k_shortest,
            weight="latency",
        )

        if len(candidate_paths) == 0:
            return None

        # 过滤掉不满足时延约束的路径
        feasible_paths = [
            path for path in candidate_paths
            if self._feasible_path(available_g, path, demand_bw, latency_limit)
        ]

        if len(feasible_paths) == 0:
            return None

        # 在前 k 条候选路径中选综合代价最优的
        best_path = min(
            feasible_paths,
            key=lambda path: self._path_score(
                p_net=g,
                path=path,
                demand_bw=demand_bw,
                prefer_rdma=rdma_only,
            )
        )
        return best_path

        # def has_enough_bw(u, v):
        #     return float(self._link_attr(g, (u, v), "bw", 0.0)) >= demand_bw

        # available_g = nx.subgraph_view(g, filter_edge=has_enough_bw)
        # try:
        #     path = nx.shortest_path(available_g, src, dst, weight="latency")
        # except (nx.NetworkXNoPath, nx.NodeNotFound):
        #     return None
        # if latency_limit is not None and self._path_latency(g, path) > latency_limit:
        #     return None
        # return path

    def _estimate_offload_gain(self, v_net, v_link) -> float:
        bw = self._vlink_bw_demand(v_net, v_link)
        msg_rate = float(self._link_attr(v_net, v_link, "msg_rate", 0.0))
        delay_sensitive = int(self._link_attr(v_net, v_link, "delay_sensitive", 0))
        flow_type = self._link_attr(v_net, v_link, "flow_type", 0)

        flow_bias = 0.0
        if flow_type in [1, "Type-1", "interactive"]:
            flow_bias = 0.30
        elif flow_type in [2, "Type-2", "sync"]:
            flow_bias = 0.15

        g_host = 0.5 * msg_rate + 0.5 * bw
        g_lat = 1.0 * delay_sensitive + flow_bias
        g_nic = 1.0

        return self.alpha_host * g_host + self.beta_lat * g_lat - self.gamma_nic * g_nic

    def _incident_rdma_gain(self, v_net, v_node_id) -> float:
        gain = 0.0
        for nb in v_net.neighbors(v_node_id):
            v_link = (v_node_id, nb) if v_net.has_edge(v_node_id, nb) else (nb, v_node_id)
            gain += max(0.0, self._estimate_offload_gain(v_net, v_link))
        return gain

    def _build_used_link_resources(self, v_net, v_link) -> Dict[str, float]:
        used = {}
        for attr in v_net.get_link_attrs(['resource']):
            used[attr.name] = float(self._link_attr(v_net, v_link, attr.name, 0.0))
        return used

    def _reserve_path(self, p_net, solution, v_net, v_link, path: List[int]) -> None:
        used_link_resources = self._build_used_link_resources(v_net, v_link)
        p_links = path_to_links(path)

        solution.link_paths[v_link] = p_links
        for p_link in p_links:
            solution.link_paths_info[(v_link, p_link)] = copy.deepcopy(used_link_resources)
            self.controller.resource_updator.update_link_resources(
                p_net, p_link, used_link_resources, operator='-'
            )

    def _consume_rdma_endpoint_budget(self, p_net, solution, v_link, p_u, p_v) -> None:
        p_net.nodes[p_u]["nic_slots"] = self._node_attr(p_net, p_u, "nic_slots", 0) - 1
        p_net.nodes[p_v]["nic_slots"] = self._node_attr(p_net, p_v, "nic_slots", 0) - 1
        for v_node, p_node in zip(v_link, (p_u, p_v)):
            slot_info = solution.node_slots_info.get((v_node, p_node))
            if slot_info is not None:
                slot_info["nic_slots"] = slot_info.get("nic_slots", 0) + 1

    # ============================================================
    # node mapping
    # ============================================================
    def node_mapping(self, v_net, p_net, solution) -> bool:
        v_net_rank = self.node_rank(v_net)
        p_net_rank = self.node_rank(p_net)

        # 先放“难点”：资源大、邻边多、通信压力大、RDMA潜在收益高
        sorted_v_nodes = list(v_net.nodes)
        sorted_v_nodes.sort(
            key=lambda v: (
                v_net_rank[v],
                self._incident_rdma_gain(v_net, v),
                v_net.degree[v],
            ),
            reverse=True
        )

        for v_node_id in sorted_v_nodes:
            selected_p_nodes = list(solution.node_slots.values())

            candidate_p_nodes = self.controller.find_candidate_nodes(
                v_net,
                p_net,
                v_node_id,
                filter=selected_p_nodes,
                check_node_constraint=True,
                check_link_constraint=False,
            )

            if len(candidate_p_nodes) == 0:
                solution['place_result'] = False
                solution['result'] = False
                return False

            best_score = float("-inf")
            best_p_node = None

            for p_node_id in candidate_p_nodes:
                # base_score = float(p_net_rank[p_node_id])
                # nic_bonus = float(self._node_attr(p_net, p_node_id, "nic_slots", 0))
                # rdma_bonus = 1.0 if self._is_rdma_node(p_net, p_node_id) else 0.0
                base_score = float(p_net_rank[p_node_id])

                if self.node_rdma_aware:
                    nic_bonus = float(self._node_attr(p_net, p_node_id, "nic_slots", 0))
                    rdma_bonus = 1.0 if self._is_rdma_node(p_net, p_node_id) else 0.0
                else:
                    nic_bonus = 0.0
                    rdma_bonus = 0.0
                feasible_neighbor_count = 0
                projected_cost = 0.0
                rdma_gain_bonus = 0.0
                infeasible = False

                
                for nb in v_net.neighbors(v_node_id):
                    if nb not in solution.node_slots:
                        continue

                    p_nb = solution.node_slots[nb]
                    v_link = (v_node_id, nb) if v_net.has_edge(v_node_id, nb) else (nb, v_node_id)

                    # demand_bw = self._vlink_bw_demand(v_net, v_link)
                    # latency_limit = self._vlink_latency_demand(v_net, v_link)

                    # tcp_path = self._choose_fast_path(
                    #     p_net=p_net,
                    #     src=p_node_id,
                    #     dst=p_nb,
                    #     demand_bw=demand_bw,
                    #     latency_limit=latency_limit,
                    #     rdma_only=False,
                    # )

                    # if tcp_path is None:
                    #     infeasible = True
                    #     break

                    # feasible_neighbor_count += 1

                    # projected_cost += self._path_score(
                    #     p_net=p_net,
                    #     path=tcp_path,
                    #     demand_bw=demand_bw,
                    #     prefer_rdma=False,
                    # )

                    # # if self._is_rdma_node(p_net, p_node_id) and self._is_rdma_node(p_net, p_nb):
                    # #     rdma_gain_bonus += max(0.0, self._estimate_offload_gain(v_net, v_link))
                    # if self.node_rdma_aware and self._is_rdma_node(p_net, p_node_id) and self._is_rdma_node(p_net, p_nb):
                    #     rdma_gain_bonus += max(0.0, self._estimate_offload_gain(v_net, v_link))
                    demand_bw = self._vlink_bw_demand(v_net, v_link)
                    latency_limit = self._vlink_latency_demand(v_net, v_link)

                    # fast TCP feasibility check for node mapping
                    tcp_path = self._choose_fast_path(
                        p_net=p_net,
                        src=p_node_id,
                        dst=p_nb,
                        demand_bw=demand_bw,
                        latency_limit=latency_limit,
                        rdma_only=False,
                    )

                    if tcp_path is None:
                        infeasible = True
                        break

                    feasible_neighbor_count += 1

                    # projected cost with slack bonus
                    slack = self._path_slack(p_net, tcp_path, demand_bw)
                    projected_cost += (
                        self._path_score(
                            p_net=p_net,
                            path=tcp_path,
                            demand_bw=demand_bw,
                            prefer_rdma=False,
                        )
                        - self.lambda_slack * slack
                    )

                    # RDMA bonus only if a real fast RDMA path exists
                    if self.node_rdma_aware and self._is_rdma_node(p_net, p_node_id) and self._is_rdma_node(p_net, p_nb):
                        fast_rdma_path = self._choose_fast_path(
                            p_net=p_net,
                            src=p_node_id,
                            dst=p_nb,
                            demand_bw=demand_bw,
                            latency_limit=latency_limit,
                            rdma_only=True,
                        )
                        if fast_rdma_path is not None:
                            rdma_gain_bonus += max(0.0, self._estimate_offload_gain(v_net, v_link))

                if infeasible:
                    continue

                score = (
                    1.0 * base_score
                    + 0.6 * nic_bonus
                    + 0.8 * rdma_bonus
                    + 2.5 * feasible_neighbor_count
                    + 0.15 * rdma_gain_bonus
                    - 1.0 * projected_cost
                )

                if score > best_score:
                    best_score = score
                    best_p_node = p_node_id

            if best_p_node is None:
                solution['place_result'] = False
                solution['result'] = False
                return False

            place_result, _ = self.controller.node_mapper.place(
                v_net, p_net, v_node_id, best_p_node, solution
            )
            if not place_result:
                solution['place_result'] = False
                solution['result'] = False
                return False

        solution['place_result'] = True
        return True
    # ============================================================
    # link mapping
    # ============================================================
    def link_mapping(self, v_net, p_net, solution) -> bool:
        if solution.get('rdma_mode', None) is None:
            solution['rdma_mode'] = OrderedDict()
        if solution.get('fallback_count', None) is None:
            solution['fallback_count'] = 0
        if solution.get('rdma_link_count', None) is None:
            solution['rdma_link_count'] = 0

        rdma_mode = cast(OrderedDict, solution['rdma_mode'])

        # 更难的链路先路由
        sorted_v_links = list(v_net.links)
        sorted_v_links.sort(
            key=lambda e: self._link_hardness(v_net, e),
            reverse=True
        )

        for v_link in sorted_v_links:
            v_u, v_v = v_link

            if v_u not in solution.node_slots or v_v not in solution.node_slots:
                solution['route_result'] = False
                solution['result'] = False
                return False

            p_u = solution.node_slots[v_u]
            p_v = solution.node_slots[v_v]

            
            # utility = self._estimate_offload_gain(v_net, v_link)
            # eligible = (
            #     self._is_rdma_node(p_net, p_u)
            #     and self._is_rdma_node(p_net, p_v)
            #     and self._has_nic_budget(p_net, p_u)
            #     and self._has_nic_budget(p_net, p_v)
            # )

            # has_rdma_candidate = any(c["mode"] == 1 for c in candidates)
            # if eligible and utility >= self.offload_threshold and not has_rdma_candidate:
            #     solution['fallback_count'] = int(solution.get('fallback_count', 0) or 0) + 1

            # # 模式比较：谁代价最低选谁
            # chosen = min(candidates, key=lambda x: x["score"])
            # chosen_path = chosen["path"]
            # chosen_mode = chosen["mode"]

            # if chosen_mode == 1:
            #     self._consume_rdma_endpoint_budget(p_net, solution, v_link, p_u, p_v)
            #     solution['rdma_link_count'] = int(solution.get('rdma_link_count', 0) or 0) + 1

            # self._reserve_path(p_net, solution, v_net, v_link, chosen_path)
            # rdma_mode[v_link] = chosen_mode
            demand_bw = self._vlink_bw_demand(v_net, v_link)
            latency_limit = self._vlink_latency_demand(v_net, v_link)
            utility = self._estimate_offload_gain(v_net, v_link)

            eligible = (
                self._is_rdma_node(p_net, p_u)
                and self._is_rdma_node(p_net, p_v)
                and self._has_nic_budget(p_net, p_u)
                and self._has_nic_budget(p_net, p_v)
            )

            chosen_path = None
            chosen_mode = 0

            # -----------------------------
            # Baseline 1: Vanilla-VNE
            # -----------------------------
            if self.mode_policy == "disabled":
                tcp_path = self._choose_best_path(
                    p_net=p_net,
                    src=p_u,
                    dst=p_v,
                    demand_bw=demand_bw,
                    latency_limit=latency_limit,
                    rdma_only=False,
                )
                if tcp_path is None:
                    solution['route_result'] = False
                    solution['result'] = False
                    return False

                chosen_path = tcp_path
                chosen_mode = 0

            # -----------------------------
            # Baseline 2: Always-Offload-If-Eligible
            # -----------------------------
            elif self.mode_policy == "always":
                rdma_path = None
                tcp_path = None

                if eligible:
                    rdma_path = self._choose_best_path(
                        p_net=p_net,
                        src=p_u,
                        dst=p_v,
                        demand_bw=demand_bw,
                        latency_limit=latency_limit,
                        rdma_only=True,
                    )

                if rdma_path is not None:
                    chosen_path = rdma_path
                    chosen_mode = 1
                else:
                    tcp_path = self._choose_best_path(
                        p_net=p_net,
                        src=p_u,
                        dst=p_v,
                        demand_bw=demand_bw,
                        latency_limit=latency_limit,
                        rdma_only=False,
                    )
                    if tcp_path is None:
                        solution['route_result'] = False
                        solution['result'] = False
                        return False

                    if eligible:
                        solution['fallback_count'] = int(solution.get('fallback_count', 0) or 0) + 1

                    chosen_path = tcp_path
                    chosen_mode = 0

            # -----------------------------
            # Ours: opportunistic
            # -----------------------------
            else:
                candidates = self._get_mode_candidates(
                    p_net=p_net,
                    v_net=v_net,
                    v_link=v_link,
                    p_u=p_u,
                    p_v=p_v,
                )

                if len(candidates) == 0:
                    solution['route_result'] = False
                    solution['result'] = False
                    return False

                tcp_candidate = next((c for c in candidates if c["mode"] == 0), None)
                rdma_candidate = next((c for c in candidates if c["mode"] == 1), None)

                # safety guard for type checker and runtime
                if tcp_candidate is None and rdma_candidate is None:
                    solution['route_result'] = False
                    solution['result'] = False
                    return False

                # no RDMA candidate at all -> TCP
                if rdma_candidate is None:
                    if tcp_candidate is None:
                        solution['route_result'] = False
                        solution['result'] = False
                        return False

                    if eligible and utility >= self.offload_threshold:
                        solution['fallback_count'] = int(solution.get('fallback_count', 0) or 0) + 1

                    chosen_path = tcp_candidate["path"]
                    chosen_mode = 0

                # no TCP candidate but RDMA exists -> use RDMA
                elif tcp_candidate is None:
                    chosen_path = rdma_candidate["path"]
                    chosen_mode = 1

                # both exist -> RDMA only if clearly better
                else:
                    tcp_score = tcp_candidate["score"]
                    rdma_score = rdma_candidate["score"]

                    if rdma_score + self.rdma_margin < tcp_score:
                        chosen_path = rdma_candidate["path"]
                        chosen_mode = 1
                    else:
                        chosen_path = tcp_candidate["path"]
                        chosen_mode = 0

            if chosen_mode == 1:
                self._consume_rdma_endpoint_budget(p_net, solution, v_link, p_u, p_v)
                solution['rdma_link_count'] = int(solution.get('rdma_link_count', 0) or 0) + 1

            self._reserve_path(p_net, solution, v_net, v_link, chosen_path)
            rdma_mode[v_link] = chosen_mode

        solution['route_result'] = True
        return True

    def _path_hops(self, path):
        return max(len(path) - 1, 0)

    def _path_score(self, p_net, path, demand_bw, prefer_rdma=False):
        """
        Lower is better.
        """
        if path is None:
            return float("inf")

        hops = self._path_hops(path)
        lat = self._path_latency(p_net, path)
        cross = self._path_cross_rack_count(p_net, path)
        bw_slack = self._path_min_bw(p_net, path) - demand_bw

        score = (
            2.0 * lat
            + 2.0 * cross
            + 1.0 * hops
            - 0.03 * bw_slack
        )
        if prefer_rdma:
            score -= 1.0
        return score

    def _link_hardness(self, v_net, v_link):
        bw = self._vlink_bw_demand(v_net, v_link)
        msg_rate = float(self._link_attr(v_net, v_link, "msg_rate", 0.0))
        delay_sensitive = int(self._link_attr(v_net, v_link, "delay_sensitive", 0))
        flow_type = self._link_attr(v_net, v_link, "flow_type", 0)

        type_bonus = 0.0
        if flow_type in [1, "Type-1", "interactive"]:
            type_bonus = 1.0
        elif flow_type in [2, "Type-2", "sync"]:
            type_bonus = 0.5

        return 2.0 * bw + 1.0 * msg_rate + 3.0 * delay_sensitive + type_bonus


    def _get_mode_candidates(self, p_net, v_net, v_link, p_u, p_v):
        """
        Return candidate routes for TCP and RDMA.
        """
        demand_bw = self._vlink_bw_demand(v_net, v_link)
        latency_limit = self._vlink_latency_demand(v_net, v_link)
        utility = self._estimate_offload_gain(v_net, v_link)

        candidates = []

        # TCP candidate
        tcp_path = self._choose_best_path(
            p_net=p_net,
            src=p_u,
            dst=p_v,
            demand_bw=demand_bw,
            latency_limit=latency_limit,
            rdma_only=False,
        )
        if tcp_path is not None:
            tcp_score = self._path_score(p_net, tcp_path, demand_bw, prefer_rdma=False)
            candidates.append({
                "mode": 0,
                "path": tcp_path,
                "score": tcp_score,
                "utility": utility,
            })

        # RDMA candidate
        eligible = (
            self._is_rdma_node(p_net, p_u)
            and self._is_rdma_node(p_net, p_v)
            and self._has_nic_budget(p_net, p_u)
            and self._has_nic_budget(p_net, p_v)
        )
        if eligible and utility >= self.offload_threshold:
            rdma_path = self._choose_best_path(
                p_net=p_net,
                src=p_u,
                dst=p_v,
                demand_bw=demand_bw,
                latency_limit=latency_limit,
                rdma_only=True,
            )
            if rdma_path is not None:
                # rdma_score = self._path_score(p_net, rdma_path, demand_bw, prefer_rdma=True)
                # # utility 越高，越倾向 RDMA
                # rdma_score -= 0.2 * utility
                rdma_score = self._path_score(
                    p_net,
                    rdma_path,
                    demand_bw,
                    prefer_rdma=True
                )
                rdma_score = (
                    rdma_score
                    - 0.2 * utility
                    + self._nic_scarcity_penalty(p_net, p_u, p_v)
                )
                candidates.append({
                    "mode": 1,
                    "path": rdma_path,
                    "score": rdma_score,
                    "utility": utility,
                })

        return candidates
    
@SolverRegistry.register(solver_name='vanilla_vne', solver_type='node_ranking')
class VanillaVNESolver(RDMARankSolver):
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        super().__init__(controller, recorder, counter, logger, config, **kwargs)
        self.node_rdma_aware = False
        self.mode_policy = "disabled"


@SolverRegistry.register(solver_name='always_offload_if_eligible', solver_type='node_ranking')
class AlwaysOffloadIfEligibleSolver(RDMARankSolver):
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        super().__init__(controller, recorder, counter, logger, config, **kwargs)
        self.node_rdma_aware = True
        self.mode_policy = "always"