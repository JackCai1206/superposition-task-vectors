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
    TO_UPPER = 'TO_UPPER'
    NOP = 'NOP'

def get_task(A,B, ops):
    assert len(ops) == 4
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
            op = OP2[op]
            if op == OP2.ADD_1:
                answer += 1
            elif op == OP2.SUB_1:
                answer -= 1
        elif i == 2:
            op = OP3[op]
            if op == OP3.TO_ENG:
                answer = num2words(answer, lang='en')
            elif op == OP3.TO_SP:
                answer = num2words(answer, lang='es')
            elif op == OP3.TO_FR:
                answer = num2words(answer, lang='fr')
        elif i == 3:
            op = OP4[op]
            if op == OP4.TO_UPPER:
                answer = answer.upper()

    return question, str(answer)

def sample_operands(k):
    assert k <= 100
    A = sample(range(0, 100), k)
    B = sample(range(0, 100), k)
    return list(zip(A, B))

def get_dataset(cfg_dict, num_examples, prompt_size):
    set_seed(42)
    dataset = []
    for i in range(num_examples):
        dist = list(cfg_dict.values())
        assert sum(dist) == 1, f'Sum of distribution is not 1: {sum(dist)}'
        tasks = list(cfg_dict.keys())
        indices = sum([[i]*int(dist[i]*prompt_size) for i in range(len(dist))], [])
        if len(indices) < prompt_size:
            indices += [0]*(prompt_size-len(indices))
        shuffle(indices)
        prompt = []
        for j, operands in enumerate(sample_operands(prompt_size)):
            prompt.append('->'.join(get_task(*operands, tasks[indices[j]].split('/'))))
        prompt = '\n'.join(prompt)

        c = 0
        while True:
            operands = sample_operands(1)[0]
            task_strs = [get_task(*operands, tasks[i].split('/')) for i in range(len(tasks))]
            questions, answers = list(map(list, zip(*task_strs)))
            if len(set(answers)) == len(answers):
                break
            c += 1
            if c > 1000:
                raise Exception('Cannot find a question with unique answers per task')
        dataset.append((prompt, '\n' + questions[0] + '->', answers))
    return dataset
