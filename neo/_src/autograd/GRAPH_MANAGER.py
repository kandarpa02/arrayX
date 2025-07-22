from typing import NamedTuple, Callable, Any

class Node:
    def __init__(self, output, parents:tuple|list, bwd_fn:Callable):
        self.output = output 
        self.parents = parents 
        self.bwd_fn = bwd_fn

    def clone_with_new_inputs(self, input_map):
        new_parents = [input_map.get(id(p), p) for p in self.parents]
        return Node(None, new_parents, self.bwd_fn)


class Tape:
    def __init__(self):
        self.nodes = []

    def add(self, node):
        self.nodes.append(node)

    def clone_template(self, original_args):
        # experimental method
        # Clones the structure of the graph but detaches values and links inputs via id.

        input_ids = {id(arg): f"input_{i}" for i, arg in enumerate(original_args)}
        template_nodes = []

        for node in self.nodes:
            cloned_node = Node(None, node.parents, node.bwd_fn)
            template_nodes.append(cloned_node)

        return TemplateTape(template_nodes, input_ids)


class TemplateTape:
    # experimental class
    def __init__(self, template_nodes, input_ids):
        self.template_nodes = template_nodes
        self.input_ids = input_ids 

    def instantiate(self, new_args):
        id_map = {}
        for idx, arg in enumerate(new_args):
            id_map[f"input_{idx}"] = arg

        real_tape = Tape()
        for t_node in self.template_nodes:
            new_parents = [id_map.get(self.input_ids.get(id(p), ""), p) for p in t_node.parents]
            out = None  # output filled later by actual op
            real_node = Node(out, new_parents, t_node.bwd_fn)
            real_tape.add(real_node)

        return real_tape


class TapeContext:
    current = None

    @classmethod
    def push(cls, tape):
        cls.current = tape
    
    @classmethod
    def pop(cls):
        cls.current=None

    @classmethod
    def add_node(cls, node:Node):
        if cls.current is not None:
            cls.current.append(node)



