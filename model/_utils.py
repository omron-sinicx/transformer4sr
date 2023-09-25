import zss
import json
import torch
import sympy
import Levenshtein

import numpy as np



#################################################
## FUNCTIONS TO TRAIN THE SYMBOLIC TRANSFORMER ##
#################################################


def count_nb_params(model, print_all=False):
    total_nb_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        nb_params = parameter.numel()
        total_nb_params += nb_params
        if print_all:
            print(name, ': ', tuple(parameter.shape), ' --> ', nb_params)
    return total_nb_params



def compute_transformer_loss(prediction, target, label_smooth=0.0):
    custom_loss = torch.nn.CrossEntropyLoss(
        ignore_index=0,  # ignore padding zeroes when compute loss
        label_smoothing=label_smooth,  # to prevent from over-fitting
    )
    loss = custom_loss(prediction, target)
    return loss



def compute_transformer_accuracy(prediction, target):
    padding_mask = (target == 0)
    correct_bool = torch.eq(torch.argmax(prediction, dim=-1), target)
    correct_bool = torch.logical_and(correct_bool, torch.logical_not(padding_mask))
    nb_correct = torch.sum(correct_bool)
    return nb_correct / torch.sum(torch.logical_not(padding_mask))



def compute_learning_rate(step, d_model, warmup=4000):
    if step==0:
        step = 1
    lr = d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    return lr



#####################################
## TRANSLATE WITH FIXED VOCABULARY ##
#####################################


# Define fixed vocabulary for everyone
MY_VOCAB = [
    'add',  # 2
    'mul',  # 3
    'sin',  # 4
    'cos',  # 5
    'log',  # 6
    'exp',  # 7
    'neg',  # 8
    'inv',  # 9
    'sqrt',  # 10
    'sq',  # 11
    'cb',  # 12
    'C',  # 13
    'x1',  # 14
    'x2',  # 15
    'x3',  # 16
    'x4',  # 17
    'x5',  # 18
    'x6',  # 19
]



def translate_integers_into_tokens(seq_int):
    seq_tokens = []
    for n in range(len(seq_int)):
        if seq_int[n]>=2:
            seq_tokens.append(MY_VOCAB[seq_int[n]-2])
    return seq_tokens



def translate_tokens_into_integers(seq_tokens):
    seq_int = []
    for token in seq_tokens:
        seq_int.append(MY_VOCAB.index(token)+2)
    return seq_int



###########################
## BEAM SEARCH FUNCTIONS ##
###########################


def is_tree_complete(seq_indices):
    """
    Check whether a given sequence of tokens defines
    a complete symbolic expression.
    """
    arity = 1
    for n in seq_indices:
        if n in [0, 1]:
            continue
            print('Predict padding or <SOS>, which is bad...')
        cur_token = MY_VOCAB[n-2]  # vocabulary is hard-coded, token 0 for padding, token 1 is <SOS>
        if cur_token in ['add', 'mul']:
            arity = arity + 2 - 1
        elif cur_token in ['sin', 'cos', 'log', 'exp', 'neg', 'inv', 'sqrt', 'sq', 'cb']:
            arity = arity + 1 - 1
        elif cur_token in ['C', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']:
            arity = arity + 0 - 1
    if arity==0:
        return True
    else:
        return False



def get_permutation_idx(idx_list, k, nb_done):
    """
    Provides the indices used to place the completed
    equations at the bottom of the list, and continue the beam search
    """
    permut_idx = []
    for i in range(k-nb_done):
        if not i in idx_list:
            permut_idx.append(int(i))
    for i in range(k-nb_done):
        if i in idx_list:
            permut_idx.append(int(i))
    for i in range(nb_done):
        permut_idx.append(int(k-nb_done+i))
    return permut_idx



def beam_search(prediction, k):
    """
    Returns the best k sequences following beam-search.
    Parameters:
      - prediction: output of the symbolic transformer.
      shape = (batch_size, seq_length, vocab_size)
      - k: Size of the beam. Corresponds to the number of
      returned sequences.
    """
    
    # Initialization
    batch_size, seq_length, vocab_size = prediction.shape
    log_prob, indices = prediction[:, 0, :].topk(k, sorted=True)
    indices = indices.unsqueeze(-1)
    nb_done = torch.zeros(batch_size, dtype=torch.int64)
    
    # Loop until end of sequence
    for n1 in range(1, seq_length):
        log_prob_temp = log_prob.unsqueeze(-1) + prediction[:, n1, :].unsqueeze(1).repeat(1, k, 1)
        
        # Prevent completed equations from generating new candidates
        for n2 in range(batch_size):
            cur_nb_done = nb_done[n2]
            for n3 in range(cur_nb_done):
                log_prob_temp[n2, -1-n3] = -1e8  # fill completed equations from the end of the table

        # Select top k equations
        log_prob_temp, index_temp = log_prob_temp.view(batch_size, -1).topk(k, sorted=True)
        new_log_prob = torch.clone(log_prob_temp)
        
        # Overwrite completed equations
        for n2 in range(batch_size):
            cur_nb_done = nb_done[n2]
            for n3 in range(cur_nb_done):
                new_log_prob[n2, -1-n3] = log_prob[n2, -1-n3]
        
        # Extract begin + concat indices
        idx_begin = index_temp // vocab_size
        idx_concat = index_temp % vocab_size

        # Generate new indices while keeping indices from completed equations untouched
        new_indices = torch.zeros((batch_size, k, n1+1), dtype=torch.int64)
        for n2 in range(batch_size):
            cur_nb_done = nb_done[n2]
            for n3 in range(k):
                if abs(n3-k) <= cur_nb_done:  # completed equations
                    new_indices[n2, n3, :-1] = indices[n2, n3]
                    new_indices[n2, n3, -1] = 0  # pad with zeros when finished
                else:
                    new_indices[n2, n3, :-1] = indices[n2][idx_begin[n2, n3]]
                    new_indices[n2, n3, -1] = idx_concat[n2, n3]
        indices = new_indices

        # Finally look for completed equation trees
        for n2 in range(batch_size):
            cur_nb_done = nb_done[n2]
            idx_list = []  # List to temporarily store completed indices
            for n3 in range(k-cur_nb_done):
                if is_tree_complete(indices[n2, n3]):
                    idx_list.append(n3)

            if len(idx_list)>0:
                permut_idx = get_permutation_idx(idx_list, k, cur_nb_done)  # List of permutation for indices
                temp = indices[n2, permut_idx]
                indices[n2] = temp
                temp = log_prob[n2, permut_idx]
                log_prob[n2] = temp
                nb_done[n2] += len(idx_list)
                    
    return indices, nb_done



def compute_solution_rate(prediction, targets, k=5):
    """
    Compute the solution rate, i.e. number of equations for which
    the ground truth is in the top-k equations returned by the
    beam-search algorithm.
    Parameters:
      - prediction: output of the symbolic transformer.
      shape = (batch_size, seq_length, vocab_size)
      - targets: ground truth for the labels.
      shape = (batch_size, seq_length)
      - k: Size of the beam for the beam-search algorithm.
      Corresponds to the number of returned sequences.
    """
    beam_indices, nb_done = beam_search(prediction, k)
    matches = torch.eq(targets.unsqueeze(1), beam_indices)
    matches = torch.all(matches, dim=-1)  # check if all tokens from the ground-truth match the equations
    matches = torch.any(matches, dim=-1)  # check if one of the top-k equations is the ground truth
    solution_rate = matches.sum() / matches.shape[0]
    return solution_rate



#####################################
## COMPUTE EDIT DISTANCE FUNCTIONS ##
#####################################


def compute_norm_levenshtein_distance(best_seq, target):
    """
    Computes Levenshtein edit distance, normalized by the length
    of the ground truth and is between [0, 1].
    Parameters:
      - best_seq: best sequence as predicted by the ST.
      Typically the result of beam_search with k=1.
      shape = (batch_size, seq_length)
      - target: ground_truth (fed to the decoder, shifted right).
      shape = (batch_size, seq_length)
    """
    norm_levenshtein_dist = torch.zeros(best_seq.shape[0])
    for n in range(best_seq.shape[0]):
        truth = translate_integers_into_equations(list(target[n].tolist()))
        pred = translate_integers_into_equations(list(best_seq[n].tolist()))
        dist = Levenshtein.distance(pred, truth)
        norm_dist = float(dist) / float(len(truth))
        norm_levenshtein_dist[n] = min(1.0, norm_dist)
    return norm_levenshtein_dist



def compute_norm_zss_distance(best_seq, target):
    """
    Computes ZSS tree edit distance, normalized by the length
    of the ground truth and is between [0, 1].
    Parameters:
      - best_seq: best sequence as predicted by the ST.
      Typically the result of beam_search with k=1.
      shape = (batch_size, seq_length)
      - target: ground_truth (fed to the decoder, shifted right).
      shape = (batch_size, seq_length)
    """
    norm_zss_dist = torch.zeros(best_seq.shape[0])
    for n in range(best_seq.shape[0]):
        truth = translate_integers_into_tokens(list(target[n].tolist()))
        pred = translate_integers_into_tokens(list(best_seq[n].tolist()))
        tree_truth = from_sequence_to_zss_tree(truth)
        tree_pred = from_sequence_to_zss_tree(pred)
        dist = zss.simple_distance(tree_truth, tree_pred)
        norm_dist = dist / float(len(truth))
        norm_zss_dist[n] = min(1.0, norm_dist)
    return norm_zss_dist



def from_sequence_to_zss_tree(seq):
    """
    Note: also works with sequences that do not correspond
    to complete equation trees!
    """
    cur_token = seq[0]
    if cur_token in ['add', 'mul']:
        split_idx = find_split_idx(seq)
        if split_idx is None:  # the equation tree has missing tokens
            tree = zss.Node(cur_token)
            if len(seq[1:])>0:
                left_kid = from_sequence_to_zss_tree(seq[1:])
                tree.addkid(left_kid)
        else:
            tree = zss.Node(cur_token)
            left_kid = from_sequence_to_zss_tree(seq[1:(split_idx+1)])
            tree.addkid(left_kid)
            if len(seq[(split_idx+1):])>0:
                right_kid = from_sequence_to_zss_tree(seq[(split_idx+1):])
                tree.addkid(right_kid)
        return tree
    elif cur_token in ['sin', 'cos', 'log', 'exp', 'neg', 'inv', 'sqrt', 'sq', 'cb']:
        tree = zss.Node(cur_token)
        if len(seq[1:])>0:
            kid = from_sequence_to_zss_tree(seq[1:])
            tree.addkid(kid)
        return tree
    elif cur_token in ['C', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']:
        leaf = zss.Node(cur_token)
        return leaf



def find_split_idx(seq):
    """
    Helper function for from_sequence_to_zss_tree.
    Locates the split index for binary nodes.
    """
    split_idx = 0
    arity = 1
    while arity>0 and (split_idx+1)<len(seq):
        split_idx += 1
        if seq[split_idx] in ['add', 'mul']:
            arity += 1
        elif seq[split_idx] in ['C', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']:
            arity += -1
    if (split_idx+1)==len(seq):
        split_idx = None
    return split_idx
