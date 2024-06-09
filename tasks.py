from enum import Enum
from functools import reduce
from random import shuffle, sample
from transformers import set_seed
from num2words import num2words

class OP1(Enum):
    ADD = 'ADD',
    SUB = 'SUB',
    COPY_A = 'COPY_A',
    COPY_B = 'COPY_B',

class OP2(Enum):
    ADD_1 = 'ADD_1',
    SUB_1 = 'SUB_1',
    NOP = 'NOP',

class OP3(Enum):
    TO_ENG = 'TO_ENG',
    TO_SP = 'TO_SP',
    TO_FR = 'TO_FR'
    NOP = 'NOP',

class OP4(Enum):
    TO_UPPER = 'TO_UPPER',
    CAP = 'CAP',
    NOP = 'NOP'

def get_task(A,B, ops):
    assert len(ops) == 4, ops
    question = f'{A}@{B}'
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
        elif i == 1:
            assert op == 'NOP' or type(answer) == int
            op = OP2[op]
            if op == OP2.ADD_1:
                answer += 1
            elif op == OP2.SUB_1:
                answer -= 1
        elif i == 2:
            assert op == 'NOP' or type(answer) == int
            op = OP3[op]
            if op == OP3.TO_ENG:
                answer = num2words(answer, lang='en')
            elif op == OP3.TO_SP:
                answer = num2words(answer, lang='es')
            elif op == OP3.TO_FR:
                answer = num2words(answer, lang='fr')
        elif i == 3:
            answer = str(answer)
            assert op == 'NOP' or not str.isdigit(answer)
            op = OP4[op]
            if op == OP4.TO_UPPER:
                answer = str.upper(answer)
            elif op == OP4.CAP:
                answer = str.capitalize(answer)

    return question, str(answer)

def sample_operands(k):
    assert k <= 100
    A = sample(range(0, 100), k)
    B = sample(range(0, 100), k)
    return list(zip(A, B))

class Dataset():
    def __init__(self, cfg_dict, num_examples, prompt_size, reseed=None):
        if reseed:
            set_seed(reseed)
            self.seed = reseed
        else:
            set_seed(42)
            self.seed = 42

        self.dist = list(cfg_dict.values())
        assert sum(self.dist) == 1, f'Sum of distribution is not 1: {sum(self.dist)}'
        self.given_tasks = list(cfg_dict.keys())
        self.given_tasks = [tuple(task.split('/')) for task in self.given_tasks]
        
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
        self.initialize_dataset()

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)

    def __eq__(self, other):
        return self.data == other.data
    
    def initialize_dataset(self):
        self.data = []
        for i in range(self.num_examples):
            indices = sum([[i]*int(self.dist[i]*self.prompt_size) for i in range(len(self.dist))], [])
            if len(indices) < self.prompt_size:
                indices += [0]*(self.prompt_size-len(indices))
            shuffle(indices)
            prompt = []
            for j, operands in enumerate(sample_operands(self.prompt_size)):
                prompt.append('->'.join(get_task(*operands, self.given_tasks[indices[j]])))
            prompt = '\n'.join(prompt)

            c = 0
            while True:
                operands = sample_operands(1)[0]
                task_strs = []
                for task in self.all_tasks:
                    try:
                        task_strs.append(get_task(*operands, task))
                    except:
                        pass
                questions, answers = tuple(map(list, zip(*task_strs)))
                if len(set(answers)) == len(answers):
                    break
                c += 1
                if c > 1000:
                    raise Exception('Cannot find a question with unique answers per task')
            self.data.append((prompt, '\n' + questions[0] + '->', answers))
