from .fol_parser import FOL_Parser
import signal

class FOL_Formula:
    def __init__(self, str_fol) -> None:
        self.parser = FOL_Parser()

        def handler(signum, frame):
            raise Exception("Timeout")

        # Set the signal handler and a 5-second alarm
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(60)

        try:
            tree = self.parser.parse_text_FOL_to_tree(str_fol)
        except Exception as exc:
            tree = None
            self.is_valid = False
            return
    
        self.tree = tree
        if tree is None:
            self.is_valid = False
        else:
            self.is_valid = True
            self.variables, self.constants, self.predicates = self.parser.symbol_resolution(tree)
    
    def __str__(self) -> str:
        _, rule_str = self.parser.msplit(''.join(self.tree.leaves()))
        return rule_str
    
    def is_valid(self):
        return self.is_valid

    def _get_formula_template(self, tree, name_mapping):
        for i, subtree in enumerate(tree):
            if isinstance(subtree, str):
                # Modify the leaf node label
                if subtree in name_mapping:
                    new_label = name_mapping[subtree]
                    tree[i] = new_label
            else:
                # Recursive call to process this subtree
                self._get_formula_template(subtree, name_mapping)

    def get_formula_template(self):
        template = self.tree.copy(deep=True)
        name_mapping = {}
        for i, f in enumerate(self.predicates):
            name_mapping[f] = 'F%d' % i
        for i, f in enumerate(self.constants):
            name_mapping[f] = 'C%d' % i

        self._get_formula_template(template, name_mapping)
        self.template = template
        _, self.template_str = self.parser.msplit(''.join(self.template.leaves()))
        return name_mapping, self.template_str
        
    
if __name__ == '__main__':
    str_fol = 'InternationalStudentIn(mike, uS)'
    fol_rule = FOL_Formula(str_fol)