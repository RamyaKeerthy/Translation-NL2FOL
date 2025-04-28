import json
from tqdm import tqdm
import argparse
from Formula import FOL_Formula

class LogicInferenceEngine:
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.dataset = self.load_logic_programs()

    def load_logic_programs(self):
        with open('../data/FOLIO/folio_train_nl_fol.json', 'r') as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} examples from {self.split} split.")
        return dataset
    
    def save_results(self, outputs):
        # Change
        with open('../data/FOLIO/folio_train_nl_fol_output.json', 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

    def safe_execute_program(self, logic_program, id):
        print(id)
        if id in ['FOLIO_train_158_453_2499', 'FOLIO_train_158_454_2505', 'FOLIO_train_158_455_2511', 'FOLIO_train_161_461_2845','FOLIO_train_161_462_2853', 'FOLIO_train_161_463_2861']:
            return "Timeout"
        program = FOL_Formula(logic_program)
        # cannot parse the program
        if program.is_valid == False:
            answer = 'Incorrect'
            return answer
        elif program.is_valid == True:
            return "Correct"
        else:
            return "Error"

    def inference_on_dataset(self):
        outputs = []
        error_count = 0
        
        for example in tqdm(self.dataset):

            # execute the logic program
            flag = self.safe_execute_program(example['FOL'].strip(), example['id'])

            if not flag == 'Correct':
                error_count += 1
            # create output
            output = {'NL': example['NL'],
                      'FOL': example['FOL'],
                      'verify': flag}
            outputs.append(output)
        
        print(f"Error count: {error_count}")
        self.save_results(outputs)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='FOLIO')
    parser.add_argument('--split', type=str, default='dev')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    engine = LogicInferenceEngine(args)
    engine.inference_on_dataset()
