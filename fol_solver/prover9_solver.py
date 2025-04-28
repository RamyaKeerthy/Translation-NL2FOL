import re
from nltk.inference.prover9 import *
from nltk.sem.logic import NegatedExpression
from .fol_prover9_parser import Prover9_FOL_Formula
from .Formula import FOL_Formula

# set the path to the prover9 executable
os.environ['PROVER9'] = '../Prover9/bin'

class FOL_Prover9_Program:
    def __init__(self, logic_program:str, dataset_name = 'FOLIO') -> None:
        self.logic_program = logic_program
        self.flag = self.parse_logic_program()
        self.dataset_name = dataset_name

    def parse_logic_program(self):
        try:        
            # Split the string into premises and conclusion
            premises_string = self.logic_program.split("Conclusion_FOL-")[0].split("Premise_FOL-")[1].strip('\n').strip(';').strip()
            conclusion_string = self.logic_program.split("Conclusion_FOL-")[1].strip('\n').strip(';').strip()

            # Extract each premise and the conclusion using regex
            premises = premises_string.strip().split(';')
            conclusion = conclusion_string.strip().split(';')

            self.logic_premises = [premise.strip() for premise in premises]
            self.logic_conclusion = conclusion[0].strip()

            # convert to prover9 format
            self.prover9_premises = []
            for premise in self.logic_premises:
                fol_rule = FOL_Formula(premise)
                if fol_rule.is_valid == False:
                    return False
                prover9_rule = Prover9_FOL_Formula(fol_rule)
                self.prover9_premises.append(prover9_rule.formula)

            fol_conclusion = FOL_Formula(self.logic_conclusion)
            if fol_conclusion.is_valid == False:
                return False
            self.prover9_conclusion = Prover9_FOL_Formula(fol_conclusion).formula
            return True
        except:
            return False

    def execute_program(self):
        try:
            goal = Expression.fromstring(self.prover9_conclusion)
            assumptions = [Expression.fromstring(a) for a in self.prover9_premises]
            timeout = 10
            #prover = Prover9()
            #result = prover.prove(goal, assumptions)
            
            prover = Prover9Command(goal, assumptions, timeout=timeout)
            result = prover.prove()
            if result:
                return 'True', ''
            else:
                # If Prover9 fails to prove, we differentiate between False and Unknown
                # by running Prover9 with the negation of the goal
                negated_goal = NegatedExpression(goal)
                prover = Prover9Command(negated_goal, assumptions, timeout=timeout)
                negation_result = prover.prove()
                if negation_result:
                    return 'False', ''
                else:
                    return 'Unknown', ''
        except Exception as e:
            print(self.prover9_conclusion)
            print(self.prover9_premises)
            return None, str(e)
        
    def answer_mapping(self, answer):
        if answer == 'True':
            return 'A'
        elif answer == 'False':
            return 'B'
        elif answer == 'Unknown':
            return 'C'
        else:
            raise Exception("Answer not recognized")
        
if __name__ == "__main__":
    logic_program = """Predicates:
        Quiet(x) ::: x is quiet
        Furry(x) ::: x is furry
        Green(x) ::: x is green
        Red(x) ::: x is red
        Rough(x) ::: x is rough
        White(x) ::: x is white
        Young(x) ::: x is young
        Premises:
        Quite(Anne) ::: Anne is quiet.
        Furry(Erin) ::: Erin is furry.
        Green(Erin) ::: Erin is green.
        Furry(Fiona) ::: Fiona is furry.
        Quite(Fiona) ::: Fiona is quiet.
        Red(Fiona) ::: Fiona is red.
        Rough(Fiona) ::: Fiona is rough.
        White(Fiona) ::: Fiona is white.
        Furry(Harry) ::: Harry is furry.
        Quite(Harry) ::: Harry is quiet.
        White(Harry) ::: Harry is white.
        Young(x) → Furry(x) ::: Young people are furry.
        Quite(Anne) → Red(x) ::: If Anne is quiet then Anne is red.
        Young(x) → Rough(x) ::: Young, green people are rough.
        Green(x) → Rough(x) ::: Young, green people are rough.
        Green(x) → White(x) ::: If someone is green then they are white.
        Furry(x) ∧ Quite(x) → White(x) ::: If someone is furry and quiet then they are white.
        Young(x) ∧ White(x) → Rough(x) ::: If someone is young and white then they are rough.
        Red(x) → Young(x) ::: All red people are young.
        Conclusion:
        Green(Fiona) ::: Fiona is green
    
    """

    prover9_program = FOL_Prover9_Program(logic_program)
    answer, error_message = prover9_program.execute_program()