import networkx as nx
import bs4
import os
import re
from scheduler.job import Job, Operator

class GPFOperator(Operator):
    GPF_node_sources_xml_template = '''
    
    '''
    def __init__(self, node: bs4.PageElement) -> None:
        super().__init__(node.attrs['id'])
        self.__bs_node = node
        self.__id = node.attrs['id']
        self.__dependencies = [ source_prod_el.attrs['refid'] for source_prod_el in node.find_all(refid=re.compile('.*')) ]
        
    def __hash__(self) -> int:
        return abs(hash(str(self.__id)))

    def __str__ (self):
        return self.id
    
    def __repr__(self):
        return f'{self.name}@{self.id}'

    def __eq__ (self, other):
        return self.__hash__() == other.__hash__()

    @property
    def dependencies(self):
        return self.__dependencies

    @property
    def id(self):
        return self.__id

    @property
    def bs_node(self):
        return self.__bs_node

    def add_source(self, refid):
        source_el = self.__bs_node.find(name='sources')
        if not source_el:
            raise Exception('"sources" container element not found in XML operator...')
            # TODO create this missing element
        new_source = bs4.BeautifulSoup().new_tag("sourceProduct", refid=refid)
        source_el.append(new_source)
        self.__dependencies.append(refid)

    def is_read (self):
        return self.bs_node.operator.getText() == "Read"
    
    def is_write (self):
        return self.bs_node.operator.getText() == "Write"


class GPFGraph(Job):
    def __init__(self, incoming_graph_data=None, xml_path: str = None) -> None:
        super().__init__(incoming_graph_data=incoming_graph_data)
        if xml_path:
            self.import_from_xml(xml_path)

    GPF_graph_xml_template = '''
    <graph id="Graph">
    <version>1.0</version>
        {nodes}
    </graph>
    '''
    GPF_read_xml_template = '''
    <node id="{nodeID}">
        <operator>Read</operator>
        <sources></sources>
        <parameters class="com.bc.ceres.binding.dom.XppDomElement">
            <file>{inputFilePath}</file>
        </parameters>
    </node>
    '''
    GPF_write_xml_template = '''
    <node id="{nodeID}">
        <operator>Write</operator>
        <sources></sources>
        <parameters class="com.bc.ceres.binding.dom.XppDomElement">
            <file>{outputFilePath}</file>
            <formatName>
                BEAM-DIMAP
            </formatName>
        </parameters>
    </node>
    '''

    def import_from_xml(self, xml_path: str) -> None:
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f'File non trovato @ {xml_path}')
        with open(xml_path, 'r') as f:
            data = f.read()
        sp = bs4.BeautifulSoup(data, 'xml')
        operators = [GPFOperator(node) for node in sp.findAll('node') ]
        for op in operators:
            self.add_node(op)
            for dep_id in op.dependencies:
                src_op = next(x for x in operators if dep_id == x.id)
                self.add_edge(src_op, op)

    def export_to_xml(self, xml_path: str) -> None:
        xml_graph = self.GPF_graph_xml_template
        nodes_xml = ''
        for node in self.nodes():
            nodes_xml = nodes_xml + str(node.bs_node)
        xml_graph = xml_graph.replace('{nodes}',  nodes_xml)
        with open(xml_path, 'w') as f:
            f.write(xml_graph)

    def add_read_source(self, input_path, node_id = 'read', skip_reads=False):
        xml_node = self.GPF_read_xml_template.replace('{inputFilePath}', input_path)
        sources = list(self.sources()).copy()
        for i,old_source in enumerate(sources):
            if skip_reads and old_source.is_read():
                continue
            actual_node_id = f'{node_id}_{i}'
            xml_node_i = xml_node.replace('{nodeID}', actual_node_id)
            bs_node = bs4.BeautifulSoup(xml_node_i, 'xml').node
            gpf_source = GPFOperator(bs_node)
            self.add_edge(gpf_source, old_source)
            old_source.add_source(gpf_source.id)
            print(f"Added {actual_node_id} as source for {gpf_source.id}")
        
    def add_write_sink(self, output_path, node_id = 'write', skip_writes=False):
        xml_node = self.GPF_write_xml_template.replace('{outputFilePath}', output_path)
        sinks = list(self.sinks()).copy()
        for i,old_sink in enumerate(sinks):
            if skip_writes and old_sink.is_write():
                continue
            actual_node_id = f'{node_id}_{i}'
            if(i > 0):
                xml_node = self.GPF_write_xml_template.replace('{outputFilePath}', f'{output_path}_{i}')
            xml_node_i = xml_node.replace('{nodeID}', actual_node_id)
            bs_node = bs4.BeautifulSoup(xml_node_i, 'xml').node
            gpf_sink = GPFOperator(bs_node)
            gpf_sink.add_source(old_sink.id)
            self.add_edge(old_sink, gpf_sink)

VISUALIZE = True
if '__main__' == __name__:
    print('--- Graph init')
    g = GPFGraph(xml_path='./test-data/sentinel2-ndvi.xml')
    print('--- Nodes')
    print(list(g.nodes))
    print('--- Edges')
    print(list(g.edges))
    # export graph
    print('--- Adding Read source node')
    g.add_read_source('/a/b/c')
    g.add_write_sink('/a/b/c')
    print('--- Adding Write sink node')
    print('--- Nodes')
    print(list(g.nodes))
    print('--- Edges')
    print(list(g.edges))
    print('--- Exporting graph')
    g.export_to_xml('./test-data/export_test.xml')
    # visualize result - NOT WORKING
    if VISUALIZE:
        import matplotlib.pyplot as plt
        print('--- Generating image...')
        fig = plt.figure()
        pos = nx.spring_layout(g)
        nx.draw(g, pos , with_labels = True, node_size=200, node_color='#fff')
        plt.show()
