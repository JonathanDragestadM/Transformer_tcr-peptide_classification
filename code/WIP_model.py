# Tranformer model for peptide-TCR binding prediction


#%%
#library
import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset


import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# import sklearn.metrics

### data load #############
# The data consit of 6 TCR chains, 1 peptide, a binary binding label and a patubation
# The data will be loaded into a list of list which entries for pabubation
# There are 5 patubations so the first list is 5 long
# in eatch entry there are a total of 7 sequence and a binding label
# The sequences will be added as "identifier - " + sequence
#    The sequence will be split by eatch AA
# bewteen eatch elemt with be ", "
  
data = []
with open("../data/nettcr_2_2_full_dataset.csv") as file:
    header = []
    data_patubation = [[],[],[],[],[]]

    for line in file:
        if header == []:
            header = line.split(",")
        else:
            content = line.split(",")
            #  header[1]+ " - " + ' '.join([x for x in content[1]]) + " "
            TCR = [header[j]+ " - " + ' '.join([x for x in content[j]]) + " " for j in range(1,7)]
            TCR_string = ", ".join(TCR)

            peptide = header[7]+ " - " + ' '.join([x for x in content[7]])

            binder_labe = content[10]

            partition = int(content[11])

            data_patubation[partition].append(", ".join([TCR_string,peptide,str(binder_labe)]))


            # if content[7] != content[12]:
            #     print(content[7] + content[12])
            # break
            # for x,y in enumerate(header):
            #     print(x,y)  







        # data.append(


# def conten_data_to_string (data):   
#     fisk = [TCR,peptide,str(binder_labe)]
#     fisk1 = " ".join(TCR)
#     fisk2 = " ".join([fisk1,peptide,binder_labe])


# split data into  train test and validate
def get_data_pabubation(data,train_index = "other", val_index = 3, test_index = 4):
    # find the train index
    train_index = list(range(0,5))
    train_index.pop(val_index)
    train_index.pop(test_index-1)

    # get train data
    train = [x for xs in [data[i] for i in train_index] for x in xs]

    val = data[val_index]  
    # val = [x for xs in data[val_index]  for x in xs]

    test = data[test_index]
    # test = [x for xs in data[test_index]  for x in xs]

    return train, val, test


def zero_pad(data_list,end_length = 198):
    zero_padded_data = []
    for entry in data_list:
        zero_to_add = int((end_length-len(entry))/2)

        zero_padded_data.append(entry[:-1]+"? "*zero_to_add+entry[-1])
    return zero_padded_data



class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, n_label: int, seq_len: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout )#, batch_first = True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        # add blossem
        self.d_model = d_model
        self.seq_len = seq_len
        self.activation = nn.Sigmoid()
        self.linear = nn.Linear(d_model , n_label)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_padding_mask: Tensor = None ) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)

        src = self.pos_encoder(src)

        #original
        # if src_mask is None:
        #     """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
        #     Unmasked positions are filled with float(0.0).
        #     """
        #     # the goal of the scr_mask if to lay forcus during on different words
        #     # generate_square_subsequent_mask, just make it a linear forcus starter from 1
        #     # then 1 and 2, then 1,2 and 3 and so on for seq_len
        #     src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(device)
        
        # output = self.transformer_encoder(src, src_padding_mask, src_mask)

        # not adding mask
        output = self.transformer_encoder(src = src, 
                                          src_key_padding_mask = src_padding_mask)
        
        # return output
        
        # mask for output size to match length
        # transpose to match dim. output is now [embed,batch, pos]
        output = output.swapaxes(0,2) * (src_padding_mask ==False)
        
        
        # output = output.T * (src_padding_mask ==False)



        # Sum acroos postion to get [embed, batch]
        output = output.sum(2)  / (src_padding_mask == False).sum(1)

        
        # transpose back to have [ batch, embed] for linear layer
        output = output.T


        output = self.linear(output)

        


        # return output



        # Code for flatten outout
        ## faltten
        #output = output.view(-1, n_label)

        # diretly pass output insted of flatten first
        # output = self.linear(output.view(-1, self.d_model*self.seq_len))
        
        
        return self.activation(output)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 93):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """


        x = x + self.pe[:x.size(0)] # x.size(0) =seq dimension, add only as much as we have sequence

        return self.dropout(x) 
    

# from https://pytorch.org/tutorials/beginner/translation_transformer.html
# create masks for seq
def create_mask(src, tgt = None):
    # scr, source. Dims [seq, batch, embed]
    src_seq_len = src.shape[0]

    # src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    # tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    # return src_mask , src_padding_mask
    return src_padding_mask


# Clean code chunk below, this is a explantiona for myself
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
# the tokennizer just transforms a sentence in to words
# In python it takes a string and splits into list, with each elemt a word
tokenizer = get_tokenizer('basic_english')
some_wards = tokenizer(data_patubation[1][1])
print(some_wards)


# map function applies a funtion to a iterable elemnt
# here the map funtion will use tokenziner on a list of all our entries for patubation 1
# Map function stores a callble funtion and does not create the obejct before called
# https://stackoverflow.com/questions/62618061/why-does-casting-pythons-map-function-change-its-effect

# list will invoke all called into none are left
alot_wards = map(tokenizer, data_patubation[1])
# print(list(alot_wards)[1])

# better way for me to undetand, just make list from start
alot_wards = list(map(tokenizer, data_patubation[1]))

alot_wards_flat = [x for xs in alot_wards for x in xs]

alot_wards_uniq = list(dict.fromkeys(alot_wards_flat))
# print(alot_wards)


# [x for xs in xss for x in xs]


# # vocab = torchtext.vocab.build_vocab_from_iterator([tokens])
# ids = [vocab[token] for token in list(map(tokenizer, data_patubation[1]))[1] ]
# print(ids)



from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
tokenizer = get_tokenizer('basic_english')

# zero pad
# find max length
list_all_data = [entry for patubation in data_patubation for entry in patubation]
max_length = max(map(len,list_all_data))


data_patubation = [zero_pad(x,max_length) for x in data_patubation]
# # zero pad
# # capture all
# zero_pad_data = []
# for pat in data_patubation:
#     # capture eatch patubation
#     pat_list = []

#     for entry in pat:
#         # zero pad entry and add to pat
#         pat_list.append(zero_pad(entry,max_length))

#     # add patubation to totalt list
#     zero_pad_data.append(pat_list)



list_all_data = [entry for patubation in data_patubation for entry in patubation]




Tokens = list(map(tokenizer,list_all_data))

# Tokens.append(tokenizer("?"))

# print(list(fisk))

PAD_IDX = 1 
specials=['<unk>',"?"]

vocab = build_vocab_from_iterator(Tokens,
                                specials=['<unk>'],
                                special_first=True)

vocab.set_default_index(vocab['<unk>'])



# %%
        
def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    # return data
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


# ``train_iter`` was "consumed" by the process of building the vocab,
# so we have to create it again

train_iter, val_iter, test_iter = get_data_pabubation(
    data_patubation,
    train_index = "other", 
    val_index = 3, 
    test_index = 4)



to_include = int(8200)

train_iter = train_iter[:to_include]
val_iter = val_iter[:to_include]  
test_iter = test_iter[:to_include]


train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %%
# original

def batchify(data: Tensor, bsz: int, seq_length = 94) -> Tensor:
    """Divides the data into ``bsz`` separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Arguments:
        data: Tensor, shape ``[N]``
        bsz: int, batch size

    Returns:
        Tensor of shape ``[seq , batches ]``
    """
    # origirnal

    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)
    

    # # this is to find the avage sentence length
    # seq_len = data.size(0) // bsz
    seq_len = seq_length
    # this just trimps the data

    data = data[:seq_len * bsz]
    
    
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)

# # original
# batch_size = 20
# eval_batch_size = 10
# train_data2 = batchify(train_data, batch_size)  # shape ``[seq_len, batch_size]``
# val_data2 = batchify(val_data, eval_batch_size)
# test_data2 = batchify(test_data, eval_batch_size)


batch_size = int(len(train_data)/94)
eval_batch_size = int(min(len(val_data),len(test_data))/94)


train_data = batchify(train_data, batch_size)  # shape ``[seq_len, batch_size]``
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)


# %%


# bptt = 94 - 1 
# def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
#     """
#     Args:
#         source: Tensor, shape ``[full_seq_len, batch_size]``
#         i: int

#     Returns:
#         tuple (data, target), where data has shape ``[seq_len, batch_size]`` and
#         target has shape ``[seq_len * batch_size]``
#     """
#     seq_len = min(bptt, len(source) - 1 - i)
#     data = source[i:i+seq_len]
#     target = source[i+1:i+1+seq_len].reshape(-1)
#     new_target = source[seq_len].reshape(-1)


#     return data, target, new_target
# what_i = 0

# batch_train_data2 = get_batch(train_data2, what_i)  # shape ``[seq_len, batch_size]``
# batch_val_data2 = get_batch(val_data2, what_i)
# batch_test_data2 = get_batch(test_data2, what_i)


bptt = 94 - 1 
def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape ``[full_seq_len, batch_size]``
        i: int

    Returns:
        returns data, target
        data has shape ``[seq_len, batch_size]`` 
        target has shape ``[label, batch_size]``
    """
    # seq_len = min(bptt, len(source) - 1 - i)
    # data = source[i:i+seq_len]
    # target = source[i+1:i+1+seq_len].reshape(-1)

    data = source[:93,i*20:20+i*20]
    target = [int(vocab.lookup_tokens([x])[0]) for x in source[93,i*20:20+i*20] ]
    target = torch.tensor(target)
    # target = source[93,i*20:20+i*20] 
    
    # train_data3.size(1)


    return data, target

# what_i = 0

# batch_train_data3 = get_batch_2(train_data3, what_i)  # shape ``[seq_len, batch_size]``
# batch_val_data3 = get_batch_2(val_data3, what_i)
# batch_test_data3 = get_batch_2(test_data3, what_i)


# %%

ntokens = len(vocab)  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 2  # number of heads in ``nn.MultiheadAttention``
dropout = 0.2  # dropout probability
n_label = 1 # number of labels to predict, this case 1 binrary
input_seq_len = 93 
batch_size = 20

model = TransformerModel(ntoken = ntokens, 
                         d_model =emsize, 
                         nhead = nhead,
                         d_hid = d_hid, 
                         nlayers = nlayers, 
                         n_label = n_label,
                         seq_len = input_seq_len, 
                         dropout = dropout).to(device)

import time

# train_data = train_data3

# loss and learning rate
# Use of BECwithlogtisLoos, binary class loss and with digits for stablity

# criterion = nn.CrossEntropyLoss()
# criterion = nn.BCELoss()
criterion = nn.BCEWithLogitsLoss()


lr = 1e-6  # learning rate

# there are a total of 22.000 entries per traning patubation
# if batch size is 20 then there are 3.300 batche in total for traning
# we set warmup to be 2/3
warm_steps = 2000 # how many step for warmup

# warmup in sub
warm_steps_sub = 600 # how many step for warmup
warm_steps = warm_steps_sub # how many step for warmup



# use socastied greadiant decent
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# define two scheuler for learning rate which are used sequentialt
# first linear 
# second step decay

# start factor is what to mutple learning rate with and total iter is step to reach max learning rate
scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, 
                                                    start_factor = 0.1,
                                                    total_iters=warm_steps)

# step size is how many step before 1 decay. Gamma is the decay size as a multiplifcal factor.
scheduler_decay = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size = 100.0,
                                                gamma=0.99)


# sequent schuculer. milestod define how many step before switching
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, 
                                                schedulers=[scheduler_warmup, scheduler_decay],
                                                milestones=[warm_steps])

# capture lr for plot
lr_list = []

def train(model: nn.Module) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()

    num_batches = train_data.size(1)// batch_size
    for batch, i in enumerate(range(0, train_data.size(1) //  batch_size)):
        data, targets = get_batch(train_data, i)
        pad_mask = create_mask(data)


        output = model(data,pad_mask)


        # get the right format
        targets = targets.double()
        output = output.double()
        output_flat = output.view(-1)

        loss = criterion(output_flat, targets)


        # targets = targets.type(torch.LongTensor)
        
        # targets = targets.view(20,1)
        # loss = criterion(output, targets+1)


        # test_target = torch.tensor([0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1], dtype=torch.long)
        # loss = criterion(output, test_target)


        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        # investe the learning rate
        scheduler.step()  
        lr_list.append(scheduler.get_last_lr()[0])



        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            print(lr)
            # mili sec  per batch
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / batch
            # cur_loss = total_loss /  log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            # total_loss = 0
            start_time = time.time()

    return total_loss/train_data.size(1) , output

def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    prediction = []
    with torch.no_grad():
        for batch, i in enumerate(range(0, eval_data.size(1) // batch_size)):
            data, targets = get_batch(eval_data, i)
            pad_mask = create_mask(data)
            output = model(data,pad_mask)

            # get the right format
            targets = targets.double()
            output = output.double()
            output_flat = output.view(-1)



            # original ? thiink they want loss scaled by words
            # output = model(data)
            # output_flat = output.view(-1, ntokens)

            # seq_len = data.size(0) # dont this tihs is right            
            # total_loss += seq_len * criterion(output_flat, targets).item()

            total_loss +=  criterion(output_flat, targets).item()


    # amount of entries  = eval_data.size()[1] 
    return total_loss / eval_data.size(1)
# %%


best_val_loss = float('inf')
epochs = 3

with TemporaryDirectory() as tempdir:
    
    total_time_start = time.time()

    best_model_params_path = "../models/temp/best_model"

    # loss for plot
    trainingEpoch_loss_list = []
    valEpoch_loss_list = []

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        # train 
        train_loss = train(model)
        trainingEpoch_loss_list.append(train_loss[0])
        
        # evalute 
        val_loss = evaluate(model, val_data)
        valEpoch_loss_list.append(val_loss)
        
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)



        if val_loss < best_val_loss:
            print("val:{}, best_val:{}".format(val_loss,best_val_loss))

            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_params_path)


        # ploting loss

        # print(trainingEpoch_loss_list,valEpoch_loss_list)


        plt.figure()
        plt.plot(range(0,epoch),trainingEpoch_loss_list, label='train_loss')
        plt.plot(range(0,epoch),valEpoch_loss_list,label='val_loss')
        plt.legend()
        plt.show()

        # scheduler.step()
        
    model.load_state_dict(torch.load(best_model_params_path)) # load best model states
# %%



torch.save(model.state_dict(), best_model_params_path)


test_loss = evaluate(model, test_data)
test_ppl = math.exp(test_loss)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.4f} | '
      f'test ppl {test_ppl:8.2f}')
print('=' * 89)

#%%
# write result 

#save the best model as a final model
final_model = "../models/end/best_model"

# loss, learning rate and time
with open("best_model_stat", "w") as outfile:
    
    # save train loss
    train_loss_as_string = ",".join(str(number) for number in trainingEpoch_loss_list)
    train_loss_as_string = "train_loss,"+train_loss_as_string

    outfile.write(train_loss_as_string)

    # save val loss
    val_loss_as_string = ",".join(str(number) for number in valEpoch_loss_list)
    val_loss_as_string = "train_loss,"+val_loss_as_string
    outfile.write(val_loss_as_string)

    # save lr
    val_loss_as_string = ",".join(str(number) for number in lr_list)
    val_loss_as_string = "train_loss,"+val_loss_as_string
    outfile.write(val_loss_as_string)



    




#%%
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


#%%
###############3 debug_learning rate #################

lr = 1e-6  # learning rate

test_poc = 5000
warm_steps = 2000

optimizer = torch.optim.SGD(model.parameters(), lr=lr)
print(get_lr(optimizer))

scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, 
                                                    start_factor = 0.1,
                                                    total_iters=warm_steps)

scheduler_decay = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size = 100.0,
                                                gamma=0.99)


scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, 
                                                schedulers=[scheduler_warmup, scheduler_decay],
                                                milestones=[warm_steps])

lr_list_warmup = []
lr_list_decay = []
lr_list = []
print(get_lr(optimizer))
print(scheduler.get_last_lr()[0])

for i in range(test_poc):
    # scheduler_warmup.step()
    # lr_warm = scheduler_warmup.get_last_lr()[0]
    # lr_list_warmup.append(lr_warm)


    # scheduler_decay.step()
    # lr_decay = scheduler_decay.get_last_lr()[0]
    # lr_list_decay.append(lr_decay)

    scheduler.step()
    lr_new = scheduler.get_last_lr()[0]
    lr_list.append(lr_new)



plt.figure()
# plt.plot(range(0,test_poc),lr_list_warmup, label='lr warmup')
# plt.plot(range(0,test_poc),lr_list_decay,label='lr Decay')

plt.plot(range(0,test_poc),lr_list,label='lr sequential')

# Setting the axis range
# plt.xlim(0, 5)  # X-axis range from 0 to 5

# plt.ylim(0, 2e-6) # Y-axis range from 0 to 20

plt.legend()
plt.show()



# %%

#### debug and test############
data, targets = get_batch(train_data, 0)
pad_mask = create_mask(data)
# seq_lenth = (pad_mask == False).sum(1)
output = model(data,pad_mask)

# get the right format
targets = targets.double()
output = output.double()
output_flat = output.view(-1)

loss = criterion(output_flat, targets)

list_out = output.T.tolist()
list_tar = targets.tolist()


vocab.lookup_indices(["?"])

#%%
###### ploting AUC ################



import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
import sklearn.metrics
y_test = list_tar  
y_score = list_out[0]

# Compute ROC curve and ROC area for each class
test_y = y_test
y_pred = y_score[0]

fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_score)#, pos_label=0)
roc_auc = sklearn.metrics.auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# %%



# mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
# mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))