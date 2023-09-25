import os
import yaml
import glob
import math
import time
import torch
import sympy
import numpy as np

from datasets._utils import from_sympy_to_sequence
from model._utils import compute_learning_rate
from model._utils import compute_transformer_loss, compute_transformer_accuracy
from model._utils import count_nb_params
from model.transformer_model import TransformerModel

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.cuda.empty_cache()  # Clear cache to free space

## ------------
## HYPER-PARAMS
## ------------

with open('config/train_symbolic_transformer.yaml', 'r') as f:
    config = yaml.safe_load(f)

PATH_DATA = config['PATH_DATA']
NB_ZFILL = config['NB_ZFILL']
PATH_OUT = config['PATH_OUT']

NB_EPOCHS = config['NB_EPOCHS']
BATCH_SIZE = config['BATCH_SIZE']
TRAIN_PROP = config['TRAIN_PROP']
VAL_PROP = config['VAL_PROP']

ENC_TYPE = config['ENC_TYPE']  # Encoder typer: one of 'mlp', 'att', or 'mix'
D_MODEL = config['D_MODEL']
H = config['H']
N_ENC = config['N_ENC']
N_DEC = config['N_DEC']
DROPOUT = config['DROPOUT']


## -------------------------
## LOAD AND PRE-PROCESS DATA
## -------------------------

nb_data = len(glob.glob(f'{PATH_DATA}/values/data_*.npy'))
print(f'\nNb datasets = {nb_data}')

data_values = []
data_tokens = []
percent = 0
for n in range(nb_data):
    if int((n+1)/nb_data*100.0) > percent:
        percent = int((n+1)/nb_data*100.0)
        print(f'{percent}% ', end='', flush=True)
        if percent%10==0:
            print('[' + time.strftime('%Y-%m-%d %H:%M:%S') + ']', flush=True)
    cur_path = f'{PATH_DATA}/values/data_{str(n).zfill(NB_ZFILL)}.npy'
    data_values.append(np.load(cur_path))
    cur_path = f'{PATH_DATA}/ground_truth/equation_{str(n).zfill(NB_ZFILL)}.txt'
    with open(cur_path) as f:
        lines = []
        for token in f.readlines():
            assert token[-1]=='\n'
            if token[0]=='C':
                lines.append('C')
            else:
                lines.append(token[:-1])
    data_tokens.append(lines)

data_values = np.array(data_values)
print(f'Shape of all datasets = {data_values.shape}')
print(f'Lenght of ground truth = {len(data_tokens)}')

# Define fixed vocabulary for everyone
MY_VOCAB = [
    'add',
    'mul',
    'sin',
    'cos',
    'log',
    'exp',
    'neg',
    'inv',
    'sqrt',
    'sq',
    'cb',
    'C',
    'x1',
    'x2',
    'x3',
    'x4',
    'x5',
    'x6',
]

# Look for maximum sequence length
max_seq_length = 0
for n in range(nb_data):
    if len(data_tokens[n])>max_seq_length:
        max_seq_length = len(data_tokens[n])
print(f'\nMax sequence length = {max_seq_length}')

vocab_size = len(MY_VOCAB)
print('Vocabulary:')
print(MY_VOCAB)
print(f'Vocab size = {vocab_size}')

data_targets = np.zeros((nb_data, max_seq_length+1))  # <SOS> until max_seq_length
for n1 in range(nb_data):
    data_targets[n1, 0] = 1  # 1 is <SOS>
    for n2 in range(len(data_tokens[n1])):
        data_targets[n1, n2+1] = MY_VOCAB.index(data_tokens[n1][n2]) + 2  # from 2 to vocab_size + 2

# Transform data into Torch tensors
torch_inputs = torch.from_numpy(data_values).unsqueeze(-1).type(torch.float32)
nb_samples = torch_inputs.shape[1]
torch_targets = torch.from_numpy(data_targets).type(torch.FloatTensor).type(torch.int64)
print(f'\nDataset input shape = {torch_inputs.shape}')
print(f'Dataset target shape = {torch_targets.shape}')
print(f'Nb samples = {nb_samples}')

# Split into {train, validation, test} sets with correct proportions
nb_obs = torch_inputs.shape[0]
idx = np.arange(nb_obs)
np.random.shuffle(idx)
train_idx = idx[:int(nb_obs*TRAIN_PROP)]
val_idx = idx[int(nb_obs*TRAIN_PROP):int(nb_obs*(TRAIN_PROP+VAL_PROP))]
test_idx = idx[int(nb_obs*(TRAIN_PROP+VAL_PROP)):]

nb_train_obs = len(train_idx)
nb_val_obs = len(val_idx)
nb_test_obs = len(test_idx)
nb_train_step_per_epoch = math.ceil(nb_train_obs / BATCH_SIZE)
nb_val_step_per_epoch = math.ceil(nb_val_obs / BATCH_SIZE)
nb_test_step = math.ceil(nb_test_obs / BATCH_SIZE)
print(f'Batch size = {BATCH_SIZE}')
print(f'Nb training steps per epoch = {nb_train_step_per_epoch}')
print(f'Nb val steps per epoch = {nb_val_step_per_epoch}')
print(f'Nb final test steps = {nb_test_step}')


## --------------------------
## BUILD SYMBOLIC TRANSFORMER
## --------------------------

symbolic_transformer = TransformerModel(
    enc_type=ENC_TYPE,
    nb_samples=nb_samples,  # Number of samples oar dataset
    max_nb_var=torch_inputs.shape[2],  # Max number of variables (including y)
    d_model=D_MODEL,
    vocab_size=vocab_size+2,  # len(vocab) + padding token + <SOS> token
    seq_length=max_seq_length,  # max_seq_length + 1 - 1 (add <SOS> but shifted right)
    h=H,
    N_enc=N_ENC,
    N_dec=N_DEC,
    dropout=DROPOUT,
)

print('')
print(f'Max number of variables = {torch_inputs.shape[2]}')
print(f'Encoder type = {ENC_TYPE}')
total_nb_params = count_nb_params(symbolic_transformer, print_all=False)
print(f'Total number params = {total_nb_params}')


## ------------------------
## CHECK WHETHER CPU OR GPU
## ------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDevice = {device}\n')
symbolic_transformer = torch.nn.DataParallel(symbolic_transformer)
symbolic_transformer.to(device)
torch_inputs = torch_inputs.to(device)
torch_targets = torch_targets.to(device)


## -------------------------------------
## OPTIMIZER AND LEARNING-RATE SCHEDULER
## -------------------------------------

optimizer = torch.optim.Adam(
    symbolic_transformer.parameters(),
    lr=1.0,
    betas=(0.9, 0.98),
    eps=1e-9,
)

lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer = optimizer,
    lr_lambda = lambda step: compute_learning_rate(step, d_model=D_MODEL, warmup=4000),
)


## -----------------------------------
## DEFINE TRAINING AND VALIDATION STEP
## -----------------------------------

def training_step(trainX, trainY, target):
    optimizer.zero_grad()
    prediction = symbolic_transformer(trainX, trainY)
    loss = compute_transformer_loss(prediction.transpose(-1, -2), target)  # transpose to compute loss
    acc = compute_transformer_accuracy(prediction, target)
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    return loss, acc

def validation_step(valX, valY, target):
    prediction = symbolic_transformer(valX, valY)
    loss = compute_transformer_loss(prediction.transpose(-1, -2), target)  # transpose to compute loss
    acc = compute_transformer_accuracy(prediction, target)
    return loss, acc
    

## ----------
## GO!! TRAIN
## ----------

list_train_loss = []
list_train_acc = []
list_val_loss = []
list_val_acc = []

print('START TRAINING! [' + time.strftime('%Y-%m-%d %H:%M:%S') + ']\n', flush=True)

for epoch in range(NB_EPOCHS):
    symbolic_transformer.train()  # activate training mode
    np.random.shuffle(train_idx)
    print(f'== Epoch {epoch+1}/{NB_EPOCHS} == ', end='')
    list_train_loss.append([])
    list_train_acc.append([])
    
    for step in range(nb_train_step_per_epoch):
        batch_idx = train_idx[(BATCH_SIZE*step):(BATCH_SIZE*(step+1))]
        trainX = torch_inputs[batch_idx]
        trainY = torch_targets[batch_idx, :-1]  # output of the decoder is shifted to the right
        target = torch_targets[batch_idx, 1:]
        loss, acc = training_step(trainX, trainY, target)
        loss, acc = training_step(trainX, trainY, target)
        list_train_loss[epoch].append(loss.to('cpu').detach().numpy())
        list_train_acc[epoch].append(acc.to('cpu').detach().numpy())
    
    symbolic_transformer.eval()  # deactivate training mode
    np.random.shuffle(val_idx)
    list_val_loss.append([])
    list_val_acc.append([])
    for step in range(nb_val_step_per_epoch):
        batch_idx = val_idx[(BATCH_SIZE*step):(BATCH_SIZE*(step+1))]
        valX = torch_inputs[batch_idx]
        valY = torch_targets[batch_idx, :-1]  # output of the decoder is shifted to the right
        target = torch_targets[batch_idx, 1:]
        loss, acc = validation_step(valX, valY, target)
        loss, acc = validation_step(valX, valY, target)
        list_val_loss[epoch].append(loss.to('cpu').detach().numpy())
        list_val_acc[epoch].append(acc.to('cpu').detach().numpy())
    
    print('[' + time.strftime('%Y-%m-%d %H:%M:%S') + '] ', flush=True)
    train_loss = np.mean(np.array(list_train_loss[epoch]))
    train_acc = np.mean(np.array(list_train_acc[epoch]))
    print(f'Train loss = {train_loss:.4f} ; Train acc = {train_acc:.4f} |', end='')
    val_loss = np.mean(np.array(list_val_loss[epoch]))
    val_acc = np.mean(np.array(list_val_acc[epoch]))
    print(f' Val loss = {val_loss:.4f} ; Val acc = {val_acc:.4f}')


## -------------
## TEST AND SAVE
## -------------

list_test_loss = []
list_test_acc = []
symbolic_transformer.eval()  # deactivate training mode
for step in range(nb_test_step):
    batch_idx = test_idx[(BATCH_SIZE*step):(BATCH_SIZE*(step+1))]
    testX = torch_inputs[batch_idx]
    testY = torch_targets[batch_idx, :-1]  # output of the decoder is shifted to the right
    target = torch_targets[batch_idx, 1:]
    loss, acc = validation_step(testX, testY, target)  # when test, use same procedure as validation
    loss, acc = validation_step(testX, testY, target)
    list_test_loss.append(loss.to('cpu').detach().numpy())
    list_test_acc.append(acc.to('cpu').detach().numpy())

test_loss = np.mean(np.array(list_test_loss))
test_acc = np.mean(np.array(list_test_acc))
print(f'\n== Testing ==')
print(f'Test loss = {test_loss:.4f} ; Test acc = {test_acc:.4f}')

print('\nSaving...')
if not os.path.exists(PATH_OUT):
    os.makedirs(PATH_OUT)
else:
    print('/!\ PATH_OUT already exists! Overwriting...')
    
np.save(PATH_OUT+'/train_loss.npy', np.array(list_train_loss))
np.save(PATH_OUT+'/train_acc.npy', np.array(list_train_acc))
np.save(PATH_OUT+'/val_loss.npy', np.array(list_val_loss))
np.save(PATH_OUT+'/val_acc.npy', np.array(list_val_acc))
np.save(PATH_OUT+'/test_loss.npy', np.array(list_test_loss))
np.save(PATH_OUT+'/test_acc.npy', np.array(list_test_acc))
torch.save(symbolic_transformer.state_dict(), PATH_OUT+'/model_weights.pt')
print(f'Finish \o/ Bye!')
