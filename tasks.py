from enum import Enum
from functools import reduce
from itertools import product
from random import shuffle, sample, randint, choice
import string
import numpy as np
from transformers import set_seed
from num2words import num2words
import torch
from hashlib import sha256
from alt_dataset_impl.generate_prompt import generate_prompt, PromptConfig, parse_raw_config_for_prompt
import yaml

continents = torch.load('country_continent_dict.pt', weights_only=True)
capitals = torch.load('country_capital_dict.pt', weights_only=True)

class OP1(Enum):
    ADD = 'ADD',
    SUB = 'SUB',
    COPY_A = 'COPY_A',
    COPY_B = 'COPY_B',
    COPY_C = 'COPY_C',
    CAPITAL = 'country_capital',
    CONTINENT = 'country_continent',
    CAPITALIZE = 'country_upper',

    locals().update({
        f'COPY_LETTER_{i}': f'COPY_LETTER_{i}' for i in range(1, 9)
    } | {
        f'ADD_SIMPLE_{i}': f'ADD_SIMPLE_{i}' for i in range(1, 9)
    })

class OP2(Enum):
    ADD_1 = 'ADD_1',
    SUB_1 = 'SUB_1',
    ADD_5 = 'ADD_5',
    SUB_5 = 'SUB_5',
    ADD_3 = 'ADD_3',
    NOP = 'NOP',

class OP3(Enum):
    TO_ENG = 'en',
    TO_ES = 'es',
    TO_FR = 'fr',
    TO_DE = 'de',
    TO_IT = 'it',
    TO_RU = 'ru',
    NOP = 'NOP',

class OP4(Enum):
    TO_UPPER = 'TO_UPPER',
    CAP = 'CAP',
    NOP = 'NOP'

def get_task_func_dict(cfg_dict):
    task_funcs_dict = {}
    for i, task in enumerate(cfg_dict.keys()):
        task = tuple(task.split('/'))
        if OP1[task[0]] in {OP1.CAPITAL, OP1.CONTINENT, OP1.CAPITALIZE}:
            task_funcs_dict[i] = {
                'name': OP1[task[0]].value[0],
                'kwargs': {
                    'symbol2': '->'
                }
            }
            question_dict = {
                'name': 'country1',
                'kwargs': {
                    'symbol2': '->'
                }
            }
        elif OP1[task[0]] in {OP1.ADD}:
            task_funcs_dict[i] = {
                'name': 'APlusB_t',
                'kwargs': {
                    'low': 10,
                    'high': 100,
                    'language': OP2[task[2]].value[0],
                    'symbol': '+',
                    'symbol2': '->'
                }
            }
            question_dict = {
                'name': 'add_translate',
                'kwargs': {
                    'low': 10,
                    'high': 100,
                    'lang_list': [OP2[t[2]].value for t in cfg_dict.keys()],
                    'symbol': '+',
                    'symbol2': '->'
                }
            }
    return task_funcs_dict, question_dict

class Dataset():
    def __init__(self, cfg_dict, num_examples, prompt_size, args, reseed=None, random_ans=None):
        if reseed is not None:
            set_seed(reseed)
            self.seed = reseed
        else:
            set_seed(42)
            self.seed = 42
        
        self.dist = list(cfg_dict.values())
        assert sum(self.dist) == 1, f'Sum of distribution is not 1: {sum(self.dist)}'
        self.given_tasks = list(cfg_dict.keys())
        self.given_tasks = [tuple(task.split('/')) for task in self.given_tasks]

        self.separator = args.separator
        
        # Gnerate the the corss-compositions of the two tasks
        self.all_tasks = set()
        for i in range(4):
            all_tasks_temp = set()
            if i == 0:
                for t in self.given_tasks:
                    all_tasks_temp.add((t[i], ))
            for t_acc in self.all_tasks:
                for t in self.given_tasks:
                    all_tasks_temp.add(tuple(list(t_acc) + [t[i]]))
            self.all_tasks = all_tasks_temp
        self.all_tasks = list(self.all_tasks)
    
        self.extra_tasks = [task for task in self.all_tasks if task not in self.given_tasks]
        self.all_tasks = self.given_tasks + self.extra_tasks

        self.cfg_dict = cfg_dict
        self.num_examples = num_examples
        self.prompt_size = prompt_size
        self.random_ans = random_ans
        self.full_range_operands = args.full_range_operands
        self.num_operands = args.num_operands
        
        if args.use_alt_dataset_impl:
            self.initialize_dataset_alt()
        else:
            self.initialize_dataset()

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)

    def __eq__(self, other):
        return self.data == other.data
    
    def sample_operands(self, k, ops_list, num_operands=2):
        if OP1[ops_list[0][0]] in {OP1.CAPITAL, OP1.CONTINENT, OP1.CAPITALIZE}:
            # The input is a country
            sample_range = list(capitals.keys())
        elif OP1[ops_list[0][0]].name.startswith('COPY_LETTER'):
            # The input is a length 8 string
            sample_range = string.ascii_letters
        elif OP1[ops_list[0][0]].name.startswith('ADD_SIMPLE'):
            # The input is a single digit number
            sample_range = range(0, 10)
        else:
            # The input is a number
            sample_range = range(10, 100)
        AB = [sample(sample_range, k=num_operands) for _ in range(k)]
        return AB
    
    def sample_operands_single(self, *ops_list):
        raise NotImplementedError('sample_operands_single is deprecated')
        A = randint(0, 100)
        B_upper_min = 100
        for ops in ops_list:
            if OP1[ops[0]] == OP1.ADD:
                if OP2[ops[1]] == OP2.ADD_1:
                    B_upper = 100 - A - 1
                elif OP2[ops[1]] == OP2.ADD_5:
                    B_upper = 100 - A - 5
                elif OP2[ops[1]] == OP2.SUB_1:
                    B_upper = A - 1
                elif OP2[ops[1]] == OP2.SUB_5:
                    B_upper = A - 5
                else:
                    B_upper = 100 - A
            elif OP1[ops[0]] == OP1.SUB:
                if OP2[ops[1]] == OP2.SUB_1:
                    B_upper = A - 1
                elif OP2[ops[1]] == OP2.SUB_5:
                    B_upper = A - 5
                elif OP2[ops[1]] == OP2.ADD_1:
                    B_upper = 100 - A + 1
                elif OP2[ops[1]] == OP2.ADD_5:
                    B_upper = 100 - A + 5
                elif OP2[ops[1]] == OP2.ADD_3:
                    B_upper = 100 - A + 3
                else:
                    B_upper = A
            else:
                B_upper = 100
            B_upper_min = min(B_upper, B_upper_min)
        B = randint(0, B_upper)
        return A, B

    def get_task(self, *operands, ops):
        A = operands[0]
        if len(operands) >= 2:
            B = operands[1]
        if len(operands) >= 3:
            C = operands[2]
        assert len(ops) == 4, ops
        question = '@'.join(map(str, operands))
        answer = None
        for i, op in enumerate(ops):
            if i == 0:
                op = OP1[op]
                if op == OP1.ADD:
                    answer = A+B
                elif op == OP1.SUB:
                    answer = A-B
                elif op == OP1.COPY_A:
                    answer = A
                elif op == OP1.COPY_B:
                    answer = B
                elif op == OP1.COPY_C:
                    answer = C
                elif op.name.startswith('COPY_LETTER'):
                    question = ''.join(map(str, operands))
                    answer = operands[int(op.name.split('_')[-1]) - 1]
                elif op.name.startswith('ADD_SIMPLE'):
                    answer = A + int(op.name.split('_')[-1])
                elif op == OP1.CAPITAL:
                    answer = capitals[A]
                elif op == OP1.CONTINENT:
                    answer = continents[A]
                elif op == OP1.CAPITALIZE:
                    answer = str.capitalize(A)
            elif i == 1:
                assert op == 'NOP' or type(answer) == int
                op = OP2[op]
                if op == OP2.ADD_1:
                    answer += 1
                elif op == OP2.SUB_1:
                    answer -= 1
                elif op == OP2.ADD_5:
                    answer += 5
                elif op == OP2.SUB_5:
                    answer -= 5
                elif op == OP2.ADD_3:
                    answer += 3
            elif i == 2:
                assert op == 'NOP' or type(answer) == int
                op = OP3[op]
                if op == OP3.TO_ENG:
                    answer = num2words(answer, lang='en')
                elif op == OP3.TO_ES:
                    answer = num2words(answer, lang='es')
                elif op == OP3.TO_FR:
                    answer = num2words(answer, lang='fr')
                elif op == OP3.TO_DE:
                    answer = num2words(answer, lang='de')
                elif op == OP3.TO_IT:
                    answer = num2words(answer, lang='it')
                elif op == OP3.TO_RU:
                    answer = num2words(answer, lang='ru')
            elif i == 3:
                answer = str(answer)
                assert op == 'NOP' or not str.isdigit(answer)
                op = OP4[op]
                if op == OP4.TO_UPPER:
                    answer = str.upper(answer)
                elif op == OP4.CAP:
                    answer = str.capitalize(answer)
        
        if self.random_ans == 'random-answers':
            answer = ''.join([choice(string.ascii_letters) for _ in range(len(answer))])
        elif self.random_ans == 'random-question-answers':
            answer = ''.join([choice(string.ascii_letters) for _ in range(len(answer))])
            question = ''.join([choice(string.ascii_letters) for _ in range(len(question))])
        elif self.random_ans == 'random-numeric-answers':
            answer = ''.join([choice(string.digits) for _ in range(len(answer))])
        elif self.random_ans is not None:
            raise Exception(f'Invalid random_ans: {self.random_ans}')

        return question, str(answer)
    
    def initialize_dataset(self):
        self.data = []
        for i in range(self.num_examples):
            indices = sum([[i]*round(self.dist[i]*self.prompt_size) for i in range(len(self.dist))], [])
            if len(indices) < self.prompt_size:
                # print(f'Warning: rounding up the number of prompts from {len(indices)} to {self.prompt_size}')
                fill_index = np.argmax(self.dist) # fill with the most common task, so no contamination for cases like 0/0/1
                indices += [fill_index]*(self.prompt_size-len(indices))
            shuffle(indices)
            assert len(set(indices)) == len([d for d in self.dist if d > 0])
            prompt = []
            if self.full_range_operands:
                for j, operands in enumerate(self.sample_operands(self.prompt_size, self.given_tasks, num_operands=self.num_operands)):
                    prompt.append(self.separator.join(self.get_task(*operands, ops=self.given_tasks[indices[j]])))
            else:
                assert self.num_operands == 2, 'Only 2 operands are supported for non-full_range_operands'
                for j in range(self.prompt_size):
                    oprations = self.given_tasks[indices[j]]
                    operands = self.sample_operands_single(oprations)
                    prompt.append(self.separator.join(self.get_task(*operands, ops=oprations)))
            prompt = '\n'.join(prompt)

            c = 0
            while True:
                # if any([OP3[task[2]] in {OP3.TO_FR, OP3.TO_DE, OP3.TO_IT, OP3.TO_RU} for task in self.all_tasks]):
                #     range_operands = (0, 10)
                # else:
                #     range_operands = (0, 100)
                if self.full_range_operands:
                    operands = self.sample_operands(1, self.all_tasks, num_operands=self.num_operands)[0]
                else:
                    operands = self.sample_operands_single(*self.all_tasks)
                task_strs = []
                for task in self.all_tasks:
                    try:
                        task_strs.append(self.get_task(*operands, ops=task))
                    except:
                        pass
                questions, answers = tuple(map(list, zip(*task_strs)))
                if len(set(answers)) == len(answers):
                    break
                c += 1
                if c > 1000:
                    raise Exception('Cannot find a question with unique answers per task')
            self.data.append((prompt, '\n' + questions[0] + self.separator, tuple(answers)))
    
    def initialize_dataset_alt(self):
        self.data = []
        task_funcs_dict, question_dict = get_task_func_dict(self.cfg_dict)
        prompt_cfg = parse_raw_config_for_prompt({
            'num_exper': self.num_examples,
            'num_examples': self.prompt_size,
            'num_tasks': len(self.cfg_dict),
            'num_tokens': 0,
            'task_funcs_dict': task_funcs_dict,
            'question': question_dict,
            'distribution': list(self.cfg_dict.values()),
            'order': 'random',
            'prompt_template_name': 'standard',
            'device': 'cuda:0',
            'seed': self.seed,
            # 'model_aliases': [
            #     'Meta-Llama-3-70B',
            #     'gpt-3.5-turbo-instruct',
            #     'Qwen1.5-72B'
            # ],
            # 'pretrained_dir': True,
            # 'dir_name': 'country1',
            # 'model_kwargs': None
            "max_tokens": 0,
            "num_beams": 1
        })
        for i in range(self.num_examples):
            prompt_question, output_choices = generate_prompt(prompt_cfg, None)
            prompt, question = prompt_question.rsplit('\n', 1)
            question = '\n' + question
            self.data.append((prompt, question, tuple(output_choices)))

    def __repr__(self):
        sorted_given_tasks, sorted_dist = tuple(zip(*sorted(list(self.cfg_dict.items()))))
        return f'Dataset({list(sorted_given_tasks)}, {list(sorted_dist)}, {self.num_examples}, {self.prompt_size}, {self.seed}, {self.random_ans}, {self.full_range_operands}, {self.num_operands})'

    def get_hash(self):
        return sha256(str(tuple(sorted(self.data))).encode()).hexdigest()
    
    simple_name_mapping = {
        "COPY_A/NOP/NOP/NOP": "copy(op1)",
        "COPY_B/NOP/NOP/NOP": "copy(op2)",
        "ADD/NOP/NOP/NOP": "op1 + op2",
        "COPY_A/SUB_1/NOP/NOP": "op1 - 1",
        "COPY_B/SUB_1/NOP/NOP": "op2 - 1",
        "COPY_B/NOP/NOP/NOP": "copy(op2)",
        "COPY_C/SUB_5/NOP/NOP": "op3 - 5",
        "COPY_A/NOP/TO_FR/NOP": "to_fr",
        "COPY_A/NOP/TO_DE/NOP": "to_de",
        "COPY_A/NOP/TO_IT/NOP": "to_it"
    }
    
    def get_simple_name(self):
        if len(self.dist) == 1:
            return self.simple_name_mapping['/'.join(self.given_tasks[0])]
        return '/'.join([f'{d:.2f}' for d in self.dist])
