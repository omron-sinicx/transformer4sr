import time
import json
import torch
import sympy
import numpy as np

from collections import OrderedDict
from model.transformer_model import TransformerModel
from model._utils import MY_VOCAB
from model._utils import count_nb_params
from model._utils import is_tree_complete
from model._utils import compute_norm_zss_distance
from model._utils import translate_integers_into_tokens, translate_tokens_into_integers
from datasets._utils import from_sympy_to_sequence, from_sequence_to_sympy, first_variables_first


# Path with model weights
MODEL_ENC_TYPE = 'mix'
LABEL_SMOOTHING = True

model_name = f'{MODEL_ENC_TYPE}_data_new_normal'
if LABEL_SMOOTHING:
    model_name += '_label_smoothing'
path_model_weights = f'../from_hixon/{model_name}/model_weights.pt'
hixon_state_dict = torch.load(path_model_weights, map_location=torch.device('cpu'))

print(f'\nMODEL: {model_name}')


# Initiate the ST (same number of parameters as hixon_state_dict weights)
symbolic_transformer = TransformerModel(
    enc_type=MODEL_ENC_TYPE,
    nb_samples=50,  # Number of samples par dataset
    max_nb_var=7,  # Max number of variables
    d_model=256,
    vocab_size=18+2,  # len(vocab) + padding token + <SOS> token
    seq_length=30,  # vocab_size + 1 - 1 (add <SOS> but shifted right)
    h=4,
    N_enc=4,
    N_dec=8,
    dropout=0.25,
)
print('')
total_nb_params = count_nb_params(symbolic_transformer, print_all=False)
print(f'Total number params = {total_nb_params}')


my_state_dict = OrderedDict()  # An OrederDict so model weights match
for key in hixon_state_dict.keys():
    assert key[:7]=="module."
    my_state_dict[key[7:]] = hixon_state_dict[key]


out = symbolic_transformer.load_state_dict(my_state_dict, strict=True)
symbolic_transformer.eval()  # deactivate training mode
print(out)


# Initiate SymPy variables
C, x1, x2, x3, x4, x5, x6 = sympy.symbols('C, x1, x2, x3, x4, x5, x6', real=True, positive=True)



def decode_with_symbolic_transformer(symbolic_transformer, dataset):
    """
    Greedy decode using ST.
    Decode until the equation tree is completed.
    Parameters:
      - symbolic_transformer: torch Module object
      - dataset: tabular dataset
      shape = (batch_size=1, nb_samples=50, nb_max_var=7, 1)
    """
    encoder_output = symbolic_transformer.encoder(dataset)  # Encoder output is fixed for the batch
    seq_length = symbolic_transformer.decoder.positional_encoding.seq_length
    decoder_output = torch.zeros((dataset.shape[0], seq_length+1), dtype=torch.int64)  # initialize Decoder output
    decoder_output[:, 0] = 1
    is_complete = torch.zeros(dataset.shape[0], dtype=torch.bool)  # check when decoding is finished
    
    for n1 in range(seq_length):
        padding_mask = torch.eq(decoder_output[:, :-1], 0).unsqueeze(1).unsqueeze(1)
        future_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        mask_dec = torch.logical_or(padding_mask, future_mask)
        temp = symbolic_transformer.decoder(
            target_seq=decoder_output[:, :-1],
            mask_dec=mask_dec,
            output_enc=encoder_output,
        )
        temp = symbolic_transformer.last_layer(temp)
        decoder_output[:, n1+1] = torch.where(is_complete, 0, torch.argmax(temp[:, n1], axis=-1))
        for n2 in range(dataset.shape[0]):
            if is_tree_complete(decoder_output[n2, 1:]):
                is_complete[n2] = True
    return decoder_output



def format_srsd_dataset(difficulty, key):
    """
    Returns tabular dataset used for prediction with the ST.
    Parameters:
      - difficulty: 'easy', 'medium', or 'hard'
      - key: dataset name, e.g. 'feynman-i.12.1'
    """
    
    # Load SRSD dataset and supp_info
    data_path = f'../../srsd_datasets/srsd-feynman_{difficulty}/test/{key}.txt'
    data = np.genfromtxt(data_path)
    path_dict = f'../../srsd_datasets/srsd-feynman_{difficulty}/supp_info.json'
    with open(path_dict, 'rb') as f:
        feynman_dict = json.load(f)
    
    # Filter valid expressions with respect to the ST (positive inputs)
    mask = np.all(data[:, :-1] > 0.0, axis=1)  # y is in the last column here
    valid_data = data[mask]
    
    # Create normalized dataset (input of ST)
    idx_rows = np.random.choice(valid_data.shape[0], 50, replace=False)
    shifts = np.zeros(valid_data.shape[1])
    new_dataset = np.zeros((50, 7))
    for k in range(valid_data.shape[1] - 1):  # y will be done separately at the end
        cur_data = valid_data[idx_rows, k]
        if feynman_dict[key]['si-derived_units'][k+1] == '$rad$':
            new_dataset[:, k+1] = cur_data
        else:
            shifts[k+1] = np.mean(np.log10(cur_data))
            new_dataset[:, k+1] = np.power(10.0, np.log10(cur_data)-shifts[k+1])
    shifts[0] = np.mean(np.log10(np.abs(valid_data[idx_rows, -1])))  # maybe some negative values for y
    signs = np.where(valid_data[idx_rows, -1]<0.0, -1.0, 1.0)
    new_dataset[:, 0] = np.power(10.0, np.log10(np.abs(valid_data[idx_rows, -1])) - shifts[0]) * signs
    
    return shifts, new_dataset



def ground_truth_srsd_datasets(sympy_expr):
    """
    Returns sequence of tokens and sequence of integers, corresponding
    to the ground truth for a given SRSD dataset.
    Parameters:
      - sympy_expr: SymPy Expression for the ground truth
    """
    sympy_expr = sympy_expr.evalf()
    srepr = sympy.srepr(sympy_expr)
    for i in range(9, 0, -1):  # i = 9, 8, ..., 2, 1
        srepr = srepr.replace(f"Symbol('x{i-1}', real=True)", f"Symbol('x{i}', real=True)")
    sympy_expr = sympy.sympify(srepr)
    target_seq_tokens = from_sympy_to_sequence(sympy_expr)
    target_seq = []
    for token in target_seq_tokens:
        target_seq.append(MY_VOCAB.index(token) + 2)
    return target_seq_tokens, target_seq



def evaluate_srsd_batch(difficulty, nb_repeat):
    """
    Evaluate the performances of the ST using all the datasets from
    the {difficulty} category.
    Catches multiple possible Exceptions for bad case scenarios.
    Parameters:
      - difficulty: one of 'easy', 'medium', or 'hard'
      - nb_repeat: number of repeat for each dataset, Int.
    """
    path = f'../../srsd_datasets/srsd-feynman_{difficulty}/supp_info.json'
    with open(path, 'rb') as f:
        feynman_dict = json.load(f)
    keys = list(feynman_dict.keys())
    keys.sort()
    
    all_zss_dist = np.zeros((len(keys), nb_repeat))
    
    for i1 in range(len(keys)):
        key = keys[i1]
        for i2 in range(nb_repeat):
            print(f'{key} => {i2+1}/{nb_repeat}', end='\r')
            try:
                shifts, new_dataset = format_srsd_dataset(difficulty=difficulty, key=key)
            except Exception as e:
                if "Cannot take a larger sample" in str(e):
                    all_zss_dist[i1, i2] = np.nan
                    continue
            encoder_input = torch.Tensor(new_dataset).unsqueeze(0).unsqueeze(-1)

            decoder_output = decode_with_symbolic_transformer(symbolic_transformer, encoder_input)
            if torch.sum(decoder_output[0, 1:])==0:  # prediction didn't work...
                all_zss_dist[i1, i2] = np.nan
                continue
            
            try:
                decoder_tokens = translate_integers_into_tokens(decoder_output[0])
                decoder_sympy = from_sequence_to_sympy(decoder_tokens)
                decoder_sympy = first_variables_first(decoder_sympy)
                decoder_sympy = sympy.simplify(sympy.factor(decoder_sympy))
                decoder_final_pred = translate_tokens_into_integers(from_sympy_to_sequence(decoder_sympy))
                decoder_final_pred = torch.Tensor(decoder_final_pred).unsqueeze(0).type(torch.int64)
            except:
                all_zss_dist[i1, i2] = np.nan
                continue

            sympy_ground_truth = sympy.sympify(feynman_dict[key]['sympy_eq_srepr']) * C  # shifted due to rescaling
            try:
                target_seq_tokens, target_seq = ground_truth_srsd_datasets(sympy_ground_truth)
            except Exception as e:
                if "is not in list" in str(e):
                    all_zss_dist[i1, i2] = np.nan
                    continue
            target_seq = torch.Tensor([target_seq]).type(torch.int64)

            norm_zss_dist = float(compute_norm_zss_distance(decoder_final_pred, target_seq))
            all_zss_dist[i1, i2] = norm_zss_dist

        mu = np.nanmean(all_zss_dist[i1])
        sigma = np.nanstd(all_zss_dist[i1])
        mini = np.nanmin(all_zss_dist[i1])
        maxi = np.nanmax(all_zss_dist[i1])
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]", end=' ')
        print(f'({i1+1}/{len(keys)})', end=' ')
        print(f'{key} => {mu:.3f} +/- {sigma:.3f}', end=' ')
        print(f'| {mini:.3f} ~ {maxi:.3f}')
    
    return all_zss_dist



NB_REPEAT = 30
LIST_DIFFICULTY = ['easy', 'medium', 'hard']

for i in range(len(LIST_DIFFICULTY)):
    cur_difficulty = LIST_DIFFICULTY[i]
    print('')
    print(f'== {cur_difficulty} ==')
    all_zss_dist = evaluate_srsd_batch(difficulty=cur_difficulty, nb_repeat=NB_REPEAT)
    print('-----------')
    print(f'=> Mean ZSS distance: {np.nanmean(all_zss_dist):.3f}', end='')
    print(f' +/- {np.nanstd(all_zss_dist):.3f}')
    print(f'=> Hit rate: {np.sum(np.any(all_zss_dist==0, axis=1))}/{all_zss_dist.shape[0]}')
    


