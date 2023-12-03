import networkx as nx 

class Operator:

    def __init__ (self, name):
        self.name = name

    def __repr__ (self):
        return self.name

    def __str__ (self):
        return self.name

    def __eq__ (self, other):
        return self.name == other.name

    def __hash__ (self):
        return hash(self.name)




class Job (nx.DiGraph):

    dag = None
    
    def __init__ (self, incoming_graph_data=None, name="job"):
        super().__init__(incoming_graph_data=incoming_graph_data, name=name)

    def sources(self):
        return self.reverse().sinks()

    def sinks(self):
        return [n for n in self.nodes if len(self.adj[n]) == 0]

    def paths(self):
        p = []
        for src in self.sources():
            for snk in self.sinks():
                p.extend(list(nx.all_simple_paths(self, src, snk)))
        return p

    def to_multidataset_job (self, N):
        """ 
        Creates a new Job as the union of N copies of this job.
        Nodes are relabeled as follows: (nodeA) -> (nodeA, 0), (nodeA, 1), ..., (nodeA,  N-1)
        """
        jobs = [type(self)(incoming_graph_data=nx.relabel_nodes(self, lambda x: (x, i))) for i in range(N)]
        return type(self)(nx.compose_all(jobs))

    def __repr__ (self):
        if len(self.nodes()) < 4:
            return repr(list(self.nodes()))
        else:
            return repr(list(self.nodes())[:3]) + "..."




