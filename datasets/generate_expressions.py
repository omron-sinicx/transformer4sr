import os
import yaml
import time
import sympy
import signal
import random
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from _utils import generate_expression
from _utils import from_sequence_to_sympy
from _utils import expression_tree_depth
from _utils import first_variables_first
from _utils import from_sympy_to_sequence
from _utils import sample_from_sympy_expression
from _utils import count_nb_variables_sympy_expr
from _utils import MY_VOCAB


with open('config/generate_expressions.yaml', 'r') as f:
    config = yaml.safe_load(f)

NB_TRAILS = config['NB_TRAILS']
NB_NODES_MIN = config['NB_NODES_MIN']
NB_NODES_MAX = config['NB_NODES_MAX']

MAX_SEC_WAIT_SIMPLIFY = config['MAX_SEC_WAIT_SIMPLIFY']
NB_NESTED_MAX = config['NB_NESTED_MAX']
NB_CONSTANTS_MIN = config['NB_CONSTANTS_MIN']
NB_CONSTANTS_MAX = config['NB_CONSTANTS_MAX']
NB_VARIABLES_MAX = config['NB_VARIABLES_MAX']
SEQ_LENGTH_MAX = config['SEQ_LENGTH_MAX']

NB_SAMPLING_PER_EQ = config['NB_SAMPLING_PER_EQ']
ORDER_OF_MAG_LIMIT = config['ORDER_OF_MAG_LIMIT']
NB_SAMPLE_PTS = config['NB_SAMPLE_PTS']
VARIABLE_TYPE = config['VARIABLE_TYPE']  # 'normal', 'log', or 'both'
PATH_OUT = config['PATH_OUT']
NB_ZFILL = config['NB_ZFILL']



print('\n' + '[' + time.strftime('%Y-%m-%d %H:%M:%S') + ']')
print('Generate a lot of expression trees...')
all_my_expr = []
percent = 0
for n in range(NB_TRAILS):
    if int((n+1)/NB_TRAILS*100.0) > percent:
        percent = int((n+1)/NB_TRAILS*100.0)
        print(f'{percent}% ', end='', flush=True)
        if percent%10==0:
            print('[' + time.strftime('%Y-%m-%d %H:%M:%S') + ']', flush=True)
    cur_expr = generate_expression(MY_VOCAB)
    all_my_expr.append(cur_expr)
print(f'Nb of expression trees generated = {NB_TRAILS}')



print('\n' + '[' + time.strftime('%Y-%m-%d %H:%M:%S') + ']')
print(f'Select expressions with more than {NB_NODES_MIN} and less than {NB_NODES_MAX} nodes...')
my_expr_filter = []  # Remove too simple and very long expressions
for n in range(len(all_my_expr)):
    if len(all_my_expr[n])>=NB_NODES_MIN and len(all_my_expr[n])<=NB_NODES_MAX:
        my_expr_filter.append(all_my_expr[n])
print(f'Nb of remaining expressions = {len(my_expr_filter)}')



def handler(signum, frame):
    raise Exception('too long')

print('\n' + '[' + time.strftime('%Y-%m-%d %H:%M:%S') + ']')
print(f'Remove invalid or very nested (>{NB_NESTED_MAX}) expressions...')
C, x1, x2, x3, x4, x5, x6 = sympy.symbols('C, x1, x2, x3, x4, x5, x6', real=True, positive=True)
nb_timeout_abort = 0
list_pb = []
my_expr_sympy = []
percent = 0
for n in range(len(my_expr_filter)):
    if int((n+1)/len(my_expr_filter)*100.0) > percent:
        percent = int((n+1)/len(my_expr_filter)*100.0)
        print(f'{percent}% ', end='', flush=True)
        if percent%10==0:
            print('[' + time.strftime('%Y-%m-%d %H:%M:%S') + ']', flush=True)
    try:
        sympy_expr = from_sequence_to_sympy(my_expr_filter[n])
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(MAX_SEC_WAIT_SIMPLIFY)
        try:
            sympy_expr = sympy.factor(sympy_expr)
            sympy_expr = sympy.simplify(sympy_expr)  # so that all expressions are represented in the same way
        except Exception as e:
            nb_timeout_abort += 1
            list_pb.append(my_expr_filter[n])
            continue
        signal.alarm(0)
        if not 'zoo' in str(sympy_expr):  # only if valid expression
            if expression_tree_depth(sympy_expr) <= NB_NESTED_MAX:  # and max tree depth is not more than NB_NESTED_MAX
                sympy_expr = first_variables_first(sympy_expr)  # log(x3)+x5 becomes log(x1)+x2
                sympy_expr = sympy.factor(sympy_expr)
                sympy_expr = sympy.simplify(sympy_expr)  # so that all expressions are represented in the same way
                if 'x1' in str(sympy_expr):  # do not include if there is no variable anymore
                    my_expr_sympy.append(sympy_expr)
    except Exception as e:
        print(n, e)
        print(my_expr_filter[n])
print(f'Remaining SymPy expressions = {len(my_expr_sympy)}')
print(f'Nb aborts because timeout: {nb_timeout_abort}')



print('\n' + '[' + time.strftime('%Y-%m-%d %H:%M:%S') + ']')
print('Clean the SymPy expression trees...')  # combine constants and rewrite powers/inverse/subtractions
print(f'Abort if Nb. const. < {NB_CONSTANTS_MIN} or Nb. const. > {NB_CONSTANTS_MAX}')
print(f'Abort is Nb. variables > {NB_VARIABLES_MAX}')
nb_pow_abort = 0
nb_const_min_abort = 0
nb_const_max_abort = 0
nb_var_max_abort = 0
nb_seqlen_abort = 0
my_expr_seq = []

for n in range(len(my_expr_sympy)):
    expr_seq = from_sympy_to_sequence(my_expr_sympy[n])
    if 'abort' in expr_seq:
        nb_pow_abort += 1
    else:
        if expr_seq.count('C') > NB_CONSTANTS_MAX:
            nb_const_max_abort += 1
        elif expr_seq.count('C') < NB_CONSTANTS_MIN:
            nb_const_min_abort += 1
        elif f'x{NB_VARIABLES_MAX+1}' in expr_seq:
            nb_var_max_abort += 1
        else:
            if len(expr_seq) > SEQ_LENGTH_MAX:
                nb_seqlen_abort += 1
            else:
                my_expr_seq.append(expr_seq)

print(f'Nb aborts because power exponent: {nb_pow_abort}')
print(f'Nb aborts because nb of constants: {nb_const_min_abort} and {nb_const_max_abort}')
print(f'Nb aborts because nb of variables: {nb_var_max_abort}')
print(f'Nb aborts because sequence length: {nb_seqlen_abort}')
print(f'=> Final number of expressions = {len(my_expr_seq)}')



temp = []
for n in range(len(my_expr_seq)):
    temp.append(str(my_expr_seq[n]))
temp = np.array(temp)
uniq, idx = np.unique(temp, return_index=True)

my_expr_uniq_seq = []
for n in idx:
    my_expr_uniq_seq.append(my_expr_seq[n])

print(f'\n** Number of unique expressions = {len(my_expr_uniq_seq)} **')



print('\n' + '[' + time.strftime('%Y-%m-%d %H:%M:%S') + ']')
print(f'Create {NB_SAMPLING_PER_EQ} datasets per equation.')
print(f'Datasets have {NB_SAMPLE_PTS} rows.')
print(f'Abort if generated value above {ORDER_OF_MAG_LIMIT:.1e}')
if not os.path.exists(f'{PATH_OUT}/ground_truth'):
    os.makedirs(f'{PATH_OUT}/ground_truth')
if not os.path.exists(f'{PATH_OUT}/values'):
    os.makedirs(f'{PATH_OUT}/values')

count_datasets = 0
nb_order_mag_abort = 0
nb_sample_pts_abort = 0
other_pbs_list = []
percent = 0

for n1 in range(len(my_expr_uniq_seq)):
    if int((n1+1)/len(my_expr_uniq_seq)*100.0) > percent:
        percent = int((n1+1)/len(my_expr_uniq_seq)*100.0)
        print(f'{percent}% ', end='', flush=True)
        if percent%10==0:
            print('[' + time.strftime('%Y-%m-%d %H:%M:%S') + ']', flush=True)
    cur_seq = my_expr_uniq_seq[n1]
    try:
        for n2 in range(NB_SAMPLING_PER_EQ):
            temp = []
            cur_gt = []  # ground truth
            for n3 in range(len(cur_seq)):
                if cur_seq[n3]=='C':
                    const_val = np.round(np.random.uniform(low=-100.0, high=100.0), decimals=2)
                    temp.append(str(const_val))
                    cur_gt.append('C=' + str(const_val))
                else:
                    temp.append(cur_seq[n3])
                    cur_gt.append(cur_seq[n3])

            try:
                cur_sympy_expr = from_sequence_to_sympy(temp)
                np_y, np_x = sample_from_sympy_expression(cur_sympy_expr, nb_samples=1000)
            except Exception as e:
                other_pbs_list.append([temp, e])
                continue

            if np.nanmax(np.abs(np_y)) > ORDER_OF_MAG_LIMIT:  # if magnitude above ORDER_OF_MAG_LIMIT, abort...
                nb_order_mag_abort += 1
            else:
                if np.sum(np.logical_not(np.isnan(np_y))) < NB_SAMPLE_PTS:  # if less than 200 pts available, abort...
                    nb_sample_pts_abort += 1
                else:
                    mask = np.logical_not(np.isnan(np_y))
                    nb_temp_obs = np.sum(mask)
                    temp_np_x = np_x[mask]
                    temp_np_y = np_y[mask]
                    my_idx = np.random.choice(nb_temp_obs, size=NB_SAMPLE_PTS, replace=False)
                    nb_var = count_nb_variables_sympy_expr(cur_sympy_expr)

                    if VARIABLE_TYPE=='normal':
                        dataset = np.zeros((NB_SAMPLE_PTS, 7))
                        dataset[:, 0] = temp_np_y[my_idx]
                        dataset[:, 1:(nb_var+1)] = temp_np_x[my_idx, :nb_var]
                    elif VARIABLE_TYPE=='log':
                        dataset = np.zeros((NB_SAMPLE_PTS, 7))
                        dataset[:, 0] = np.log(np.abs(temp_np_y[my_idx])+1e-10)
                        dataset[:, 1:(nb_var+1)] = np.log(np.abs(temp_np_x[my_idx, :nb_var])+1e-10)
                    elif VARIABLE_TYPE=='both':
                        dataset = np.zeros((NB_SAMPLE_PTS, 14))
                        dataset[:, 0] = temp_np_y[my_idx]
                        dataset[:, 1] = np.log(np.abs(temp_np_y[my_idx])+1e-10)
                        dataset[:, 2:(2*nb_var+1):2] = temp_np_x[my_idx, :nb_var]
                        dataset[:, 3:(2*nb_var+2):2] = np.log(np.abs(temp_np_x[my_idx, :nb_var])+1e-10)
                    else:
                        print("VARIABLE_TYPE should be one of 'normal', 'log', or 'both'")

                    np.save(f'{PATH_OUT}/values/data_{str(count_datasets).zfill(NB_ZFILL)}.npy', dataset)
                    with open(f'{PATH_OUT}/ground_truth/equation_{str(count_datasets).zfill(NB_ZFILL)}.txt', 'w') as f:
                        for token in cur_gt:
                            f.write(f'{token}\n')
                    count_datasets += 1
    except Exception as e:
        print(n1, e)
        print(cur_seq)

print(' '*100, flush=True)
print(f'Nb aborts because magnitude: {nb_order_mag_abort}')
print(f'Nb aborts because sample points: {nb_sample_pts_abort}')
print(f'Nb aborts other reasons: {len(other_pbs_list)}')
print(f'=> NUMBER OF DATASETS CREATED = {count_datasets}')

print(f'\nCheck that {nb_order_mag_abort} + {nb_sample_pts_abort} + ', end='')
print(f'{len(other_pbs_list)*NB_SAMPLING_PER_EQ} + {count_datasets} = ', end='')
print(f'{nb_order_mag_abort+nb_sample_pts_abort+len(other_pbs_list)+count_datasets}', end='')
print(f' and {NB_SAMPLING_PER_EQ} * {len(my_expr_uniq_seq)} = {NB_SAMPLING_PER_EQ*len(my_expr_uniq_seq)}')

print('\nPrint other problems:')
for pb in other_pbs_list:
    print(pb)
print('\n' + '[' + time.strftime('%Y-%m-%d %H:%M:%S') + ']')
print('Finish!')
