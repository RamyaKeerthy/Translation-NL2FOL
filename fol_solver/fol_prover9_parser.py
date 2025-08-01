from ply import lex, yacc
from .Formula import FOL_Formula

"""
Revised grammar v2:
S -> F | QUANT VAR S | '¬' S | '('QUANT VAR S')'
F -> '¬' '(' F ')' | '(' F ')' | F OP F | L 
OP -> '⊕' | '∨' | '∧' | '→' | '↔'
L -> '¬' PRED '(' TERMS ')' | PRED '(' TERMS ')'
TERMS -> TERM | TERM ',' TERMS
TERM -> CONST | VAR
QUANT -> '∀' | '∃'
"""

class Prover9_FOL_Formula:
    def __init__(self, fol_formula : FOL_Formula) -> None:
        self.tokens = ['QUANT', 'VAR', 'NOT', 'LPAREN', 'RPAREN', 'OP', 'PRED', 'COMMA', 'CONST']

        self.t_QUANT = r'∀|∃'
        self.t_NOT = r'¬'
        self.t_LPAREN = r'\('
        self.t_RPAREN = r'\)'
        self.t_OP = r'⊕|∨|∧|→|↔'
        self.t_COMMA = r','

        if len(fol_formula.variables) > 0:
            self.t_VAR = r'|'.join(list(fol_formula.variables))
        else:
            self.t_VAR = r'x'
        
        if len(fol_formula.predicates) > 0:
            self.t_PRED = r'|'.join(list(fol_formula.predicates))
        else:
            self.t_PRED = r'PRED'

        if len(fol_formula.constants) > 0:
            self.t_CONST = r'|'.join(list(fol_formula.constants))
        else:
            self.t_CONST = r'0'

        self.precedence = (
            ('left', 'OP'),
            ('right', 'NOT'),
        )

        self.t_ignore = ' \t'
        self.lexer = lex.lex(module=self)
        self.parser = yacc.yacc(module=self, write_tables=False, debug=False)
        
        self.formula = self.parse(str(fol_formula))

    def t_error(self, t):
        t.lexer.skip(1)

    # S -> F
    def p_S_F(self, p):
        '''expr : F'''
        p[0] = p[1]

    # S -> QUANT VAR S
    def p_S_quantified_S(self, p):
        '''expr : QUANT VAR expr'''
        if p[1] == "∀":
            p[0] = f"all {p[2]}.({p[3]})"
        elif p[1] == "∃":
            p[0] = f"some {p[2]}.({p[3]})"

    #Added
    # S -> '(' QUANT VAR S ')'
    def p_S_paren_quantified_S(self, p):
        '''expr : LPAREN QUANT VAR expr RPAREN'''
        if p[2] == "∀":
            p[0] = f"all {p[3]}.({p[4]})"
        elif p[2] == "∃":
            p[0] = f"some {p[3]}.({p[4]})"
    
    # S -> '¬' S
    def p_S_not(self, p):
        '''expr : NOT expr'''
        p[0] = f"not ({p[2]})"

    # F -> '¬' '(' F ')'
    def p_F_not(self, p):
        '''F : NOT LPAREN F RPAREN'''
        p[0] = f"not ({p[3]})"

    # F -> '(' F ')'
    def p_F_paren(self, p):
        '''F : LPAREN F RPAREN'''
        p[0] = p[2]

    # F -> Var
    def p_F_var(self, p):
        '''F : VAR'''
        p[0] = p[1]

    # F -> F OP F
    def p_F_op(self, p):
        '''F : F OP F'''
        if p[2] == "⊕":
            p[0] = f"(({p[1]}) & not ({p[3]})) | (not ({p[1]}) & ({p[3]}))"
        elif p[2] == "∨":
            p[0] = f"({p[1]}) | ({p[3]})"
        elif p[2] == "∧":
            p[0] = f"({p[1]}) & ({p[3]})"
        elif p[2] == "→":
            p[0] = f"({p[1]}) -> ({p[3]})"
        elif p[2] == "↔":
            p[0] = f"({p[1]}) <-> ({p[3]})"

    # F -> L
    def p_F_L(self, p):
        '''F : L'''
        p[0] = p[1]

    # L -> '¬' PRED '(' TERMS ')'
    def p_L_not(self, p):
        '''L : NOT PRED LPAREN TERMS RPAREN'''
        p[0] = f"not {p[2]}({p[4]})"

    # L -> PRED '(' TERMS ')'
    def p_L_pred(self, p):
        '''L : PRED LPAREN TERMS RPAREN'''
        p[0] = f"{p[1]}({p[3]})"

    # TERMS -> TERM
    def p_TERMS_TERM(self, p):
        '''TERMS : TERM'''
        p[0] = p[1]

    # TERMS -> TERM ',' TERMS
    def p_TERMS_TERM_TERMS(self, p):
        '''TERMS : TERM COMMA TERMS'''
        p[0] = f"{p[1]}, {p[3]}"

    # TERM -> CONST
    def p_TERM_CONST(self, p):
        '''TERM : CONST'''
        p[0] = p[1]

    # TERM -> VAR
    def p_TERM_VAR(self, p):
        '''TERM : VAR'''
        p[0] = p[1]

    def p_error(self, p):
        # print("Syntax error at '%s'" % p.value)
        pass

    def parse(self, s):
        return self.parser.parse(s, lexer=self.lexer)

if __name__ == "__main__":
    str_fol = '∀x (Male(x) ∧ TennisPlayer(x) ∧ At(x, rolandGarros2022) → LostTo(x, świątek) ∧ At(x, rolandGarros2022))'
    fol_rule = FOL_Formula(str_fol) # formula in its true form
    prover9_rule = Prover9_FOL_Formula(fol_rule)