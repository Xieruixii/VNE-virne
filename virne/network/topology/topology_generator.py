import networkx as nx


class TopologyGenerator:
    """
    Utility class to generate various networkx topologies for BaseNetwork.
    """
    @staticmethod
    def generate(type: str, num_nodes: int, **kwargs) -> nx.Graph:
        assert num_nodes >= 1, "num_nodes must be >= 1."
        match type:
            case 'path':
                return nx.path_graph(num_nodes)
            case 'star':
                return nx.star_graph(num_nodes - 1)
            case 'grid_2d':
                m = kwargs.get('m')
                n = kwargs.get('n')
                if m is None or n is None:
                    raise ValueError("'grid_2d' type requires 'm' and 'n' keyword arguments.")
                return nx.grid_2d_graph(m, n, periodic=False)
            case 'waxman':
                wm_alpha = kwargs.get('wm_alpha', 0.5)
                wm_beta = kwargs.get('wm_beta', 0.2)
                not_connected = True
                while not_connected:
                    G = nx.waxman_graph(num_nodes, wm_alpha, wm_beta)
                    not_connected = not nx.is_connected(G)
                return G
            case 'random':
                random_prob = kwargs.get('random_prob', 0.5)
                not_connected = True
                while not_connected:
                    G = nx.erdos_renyi_graph(num_nodes, random_prob, directed=False)
                    not_connected = not nx.is_connected(G)
                return G
            case 'fat_tree':
                k = int(kwargs.get('k', 8))
                if k < 2 or k % 2 != 0:
                    raise ValueError("'fat_tree' type requires an even 'k' >= 2.")
                half_k = k // 2
                G = nx.Graph()
                node_id = 0
                edge_switches = {}
                agg_switches = {}
                core_switches = {}

                # Pod switches
                for pod_id in range(k):
                    for edge_idx in range(half_k):
                        rack_id = pod_id * half_k + edge_idx
                        G.add_node(node_id, layer='edge', pod_id=pod_id, rack_id=rack_id)
                        edge_switches[(pod_id, edge_idx)] = node_id
                        node_id += 1
                    for agg_idx in range(half_k):
                        G.add_node(node_id, layer='agg', pod_id=pod_id, rack_id=-1)
                        agg_switches[(pod_id, agg_idx)] = node_id
                        node_id += 1

                # Core switches
                for group_idx in range(half_k):
                    for core_idx in range(half_k):
                        G.add_node(node_id, layer='core', pod_id=-1, rack_id=-1)
                        core_switches[(group_idx, core_idx)] = node_id
                        node_id += 1

                # Servers under edge switches
                for pod_id in range(k):
                    for edge_idx in range(half_k):
                        edge_node = edge_switches[(pod_id, edge_idx)]
                        rack_id = G.nodes[edge_node]['rack_id']
                        for _ in range(half_k):
                            G.add_node(node_id, layer='server', pod_id=pod_id, rack_id=rack_id)
                            G.add_edge(edge_node, node_id)
                            node_id += 1

                # Pod-level switch links (edge <-> agg)
                for pod_id in range(k):
                    for edge_idx in range(half_k):
                        for agg_idx in range(half_k):
                            G.add_edge(edge_switches[(pod_id, edge_idx)], agg_switches[(pod_id, agg_idx)])

                # Aggregation <-> core links
                for pod_id in range(k):
                    for agg_idx in range(half_k):
                        agg_node = agg_switches[(pod_id, agg_idx)]
                        for core_idx in range(half_k):
                            G.add_edge(agg_node, core_switches[(agg_idx, core_idx)])
                return G
            case _:
                raise NotImplementedError(f"Graph type '{type}' is not implemented.")
