#!/usr/bin/env python
# coding: utf-8

# In[16]:


import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler

torch.manual_seed(0)
np.random.seed(0)

# This concept is also called teacher forceing. 
# The flag decides if the loss will be calculted over all 
# or just the predicted values.
calculate_loss_over_all_values = False

# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number

#src = torch.rand((10, 32, 512)) # (S,N,E) 
#tgt = torch.rand((20, 32, 512)) # (T,N,E)
#out = transformer_model(src, tgt)
#
#print(out)
root_dir = r'final data/'

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
       

class TransAm(nn.Module):
    def __init__(self,feature_size=250,num_layers=1,dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)#, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask



# if window is 100 and prediction step is 1
# in -> [0..99]
# target -> [1..100]
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = np.append(input_data[i:i+tw][:-output_window] , output_window * [0])
        train_label = input_data[i:i+tw]
        #train_label = input_data[i+output_window:i+tw+output_window]
        inout_seq.append((train_seq ,train_label))
    return torch.FloatTensor(inout_seq)
def get_batch(source, i,batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i+seq_len]    
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window,1)) # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window,1))
    return input, target


def train(train_data):
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i,batch_size)
        optimizer.zero_grad()
        output = model(data)        

        if calculate_loss_over_all_values:
            loss = criterion(output, targets)
        else:
            loss = criterion(output[-output_window:], targets[-output_window:])
    
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
            
def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i,eval_batch_size)
            output = eval_model(data)            
            if calculate_loss_over_all_values:
                total_loss += len(data[0])* criterion(output, targets).cpu().item()
            else:                                
                total_loss += len(data[0])* criterion(output[-output_window:], targets[-output_window:]).cpu().item()            
    return total_loss / len(data_source)

##for displaying the plot in notebook

def predict_future_show(eval_model, data_source,steps):
    eval_model.eval() 
    total_loss = 0.
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)
    _ , data = get_batch(data_source, 0,1)
    with torch.no_grad():
        for i in range(0, steps,1):
            input = torch.clone(data[-input_window:])
            input[-output_window:] = 0     
            output = eval_model(data[-input_window:])                        
            data = torch.cat((data, output[-1:]))
            
    data = data.cpu().view(-1)
    

    pyplot.plot(data,color="red")       
    pyplot.plot(data[:input_window],color="blue")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.show()
    pyplot.close()
    return data
    
def predict_future(eval_model, data_source,steps):
    eval_model.eval() 
    total_loss = 0.
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)
    _ , data = get_batch(data_source, 0,1)
    with torch.no_grad():
        for i in range(0, steps,1):
            input = torch.clone(data[-input_window:])
            input[-output_window:] = 0     
            output = eval_model(data[-input_window:])                        
            data = torch.cat((data, output[-1:]))
            
    data = data.cpu().view(-1)
    

    pyplot.plot(data,color="red")       
    pyplot.plot(data[:input_window],color="blue")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.savefig('graph_stock/transformer-future%d.png'%steps)
    pyplot.close()
    
def plot_and_loss(eval_model, data_source,epoch):
    eval_model.eval() 
    total_loss = 0.
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i,1)
            # look like the model returns static values for the output window
            output = eval_model(data)    
            if calculate_loss_over_all_values:                                
                total_loss += criterion(output, target).item()
            else:
                total_loss += criterion(output[-output_window:], target[-output_window:]).item()
            
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0) #todo: check this. -> looks good to me
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)
            
    #test_result = test_result.cpu().numpy()
    len(test_result)

    pyplot.plot(test_result,color="red")
    pyplot.plot(truth[:500],color="blue")
    pyplot.plot(test_result-truth,color="green")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.savefig('graph_stock/transformer-epoch%d.png'%epoch)
    pyplot.close()
    
    return total_loss / i
    
def plot_and_loss_show(eval_model, data_source,epoch):
    eval_model.eval() 
    total_loss = 0.
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i,1)
            # look like the model returns static values for the output window
            output = eval_model(data)    
            if calculate_loss_over_all_values:                                
                total_loss += criterion(output, target).item()
            else:
                total_loss += criterion(output[-output_window:], target[-output_window:]).item()
            
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0) #todo: check this. -> looks good to me
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)
            
    #test_result = test_result.cpu().numpy()
    len(test_result)

    pyplot.plot(test_result,color="red")
    pyplot.plot(truth[:500],color="blue")
    pyplot.plot(test_result-truth,color="green")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.show()
    pyplot.close()
    
    return total_loss / i

def movement_from_price(np_array):
    prev_day_price = np.array(np_array)
    next_day_price = prev_day_price[1:]
    prev_day_price = prev_day_price[:-1]
    price_diff = np.subtract(next_day_price,prev_day_price)
    diff_percent = np.divide(price_diff,prev_day_price)*100
    return diff_percent
def lognormal_from_price(np_array):
    stock_price = np.array(np_array)
    log_stock_price = np.log(stock_price)
    return log_stock_price

def price_from_lognormal(np_array):
    return np.exp(np.array(np_array))

def price_from_movement(np_array,first_day_price):
    new_day_price = np.zeros(np_array.shape)
    prev_price = first_day_price
    for i in range(len(np_array)):
        new_day_price[i] = prev_price+prev_price*np_array[i]/100
        prev_price = new_day_price[i]
    first_day_array = np.reshape(np.array(first_day_price),-1)
    #print(new_day_price.shape)
    #print(first_day_array)
    final_result = np.concatenate((first_day_array,new_day_price))
    return final_result


# In[17]:


import pandas as pd
from os import listdir
from os.path import isfile, join



def load_csv_to_map(root_dir):
    all_stock_files = [f for f in listdir(root_dir) if isfile(join(root_dir, f))]
    print(listdir(root_dir))
    df_map = dict()
    for i in all_stock_files:
        current_df = pd.read_csv(root_dir+i)
        name = current_df['Symbol'][0]
        df_map[name] = current_df
    return df_map





# In[20]:


def transform_stock_data(input_stock_price):
    #this function is for transforming stock data 
    #we first take log of the data and use minmaxscaler to make the max and the min of the price
    #fit between 0 and 1
    log_price = lognormal_from_price(np.array(input_stock_price))
    scaler = MinMaxScaler(feature_range=(-1, 1)) 
    scaled_price = scaler.fit_transform(log_price.reshape(-1, 1)).reshape(-1)
    return scaler,scaled_price

def inverse_transform_predict_data(scaler,output_sequence):
    transform_back = scaler.inverse_transform(output_sequence.reshape(-1, 1)).reshape(-1)
    exp_price = price_from_lognormal(transform_back)
    return exp_price


# In[21]:





def get_stock_data(input_sequence):
    time        = np.arange(0, len(input_sequence)-1, 1) #total 2515 days
    stock_price = np.array(input_sequence)
    training_idx_start = 0
    training_idx_end = 2520
    test_idx_start = 5
    test_idx_end = 2525
    stock_price = stock_price[training_idx_start:test_idx_end]
    
    #from pandas import read_csv
    #series = read_csv('daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
    
    
    #scaler = MinMaxScaler(feature_range=(-1, 1)) 
    #amplitude = scaler.fit_transform(series.to_numpy().reshape(-1, 1)).reshape(-1)
    #stock_price = scaler.fit_transform(stock_price.reshape(-1, 1)).reshape(-1)
    
    #use first 1250 days to predict 1250 to 2100 day
    
    train_data = stock_price[training_idx_start:training_idx_end]
    test_data = stock_price[test_idx_start:test_idx_end]

    # convert our train data into a pytorch train tensor
    # train_tensor = torch.FloatTensor(train_data).view(-1)
    # todo: add comment.. 
    train_sequence = create_inout_sequences(train_data,input_window)
    train_sequence = train_sequence[:-output_window] #todo: fix hack?

    #test_data = torch.FloatTensor(test_data).view(-1) 
    test_data = create_inout_sequences(test_data,input_window)
    test_data = test_data[:-output_window] #todo: fix hack?

    return train_sequence.to(device),test_data.to(device)

def get_stock_data_prediction_only(input_sequence):
    time        = np.arange(0, len(input_sequence)-1, 1) #total 2515 days
    stock_price = np.array(input_sequence)
    training_idx_start = 0
    training_idx_end = 2520
    pred_idx_start = len(input_sequence)-INPUT_WINDOW_SIZE-PADDING_SIZE
    pred_idx_end = len(input_sequence)
    stock_price = stock_price[training_idx_start:pred_idx_end]
    
    #from pandas import read_csv
    #series = read_csv('daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
    
    
    #scaler = MinMaxScaler(feature_range=(-1, 1)) 
    #amplitude = scaler.fit_transform(series.to_numpy().reshape(-1, 1)).reshape(-1)
    #stock_price = scaler.fit_transform(stock_price.reshape(-1, 1)).reshape(-1)
    
    #use first 1250 days to predict 1250 to 2100 day
    
    train_data = stock_price[training_idx_start:training_idx_end]
    pred_data = stock_price[pred_idx_start:pred_idx_end]
    print(pred_data.shape)
    # convert our train data into a pytorch train tensor
    # train_tensor = torch.FloatTensor(train_data).view(-1)
    # todo: add comment.. 
    train_sequence = create_inout_sequences(train_data,input_window)
    train_sequence = train_sequence[:-output_window] #todo: fix hack?

    #test_data = torch.FloatTensor(test_data).view(-1) 
    pred_data = create_inout_sequences(pred_data,input_window)
    pred_data = pred_data[:-output_window] #todo: fix hack?
    print(pred_data.shape)
    return train_sequence.to(device),pred_data.to(device)








df_table = load_csv_to_map(root_dir)


input_stock_price_sequence = df_table['VZ']["Close"]#change your input stock here

INPUT_WINDOW_SIZE = 240
PADDING_SIZE = 50
NUM_EPOCHS = 300
input_window = INPUT_WINDOW_SIZE
output_window = 5
batch_size = 240 # batch size #非常重要 如果batch size 太小不会converge
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#set scaler for inverse transform
scaler,scaled_price = transform_stock_data(input_stock_price_sequence)
scaled_price_modified_for_prediction = np.concatenate((scaled_price,np.repeat(scaled_price[-1],PADDING_SIZE)))
train_data, val_data = get_stock_data(scaled_price)
_,pred_data = get_stock_data_prediction_only(scaled_price_modified_for_prediction)
model = TransAm().to(device)

criterion = nn.MSELoss()
lr = 0.005 
#optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)

best_val_loss = float("inf")
epochs = 200 # The number of epochs
best_model = None
loss_list = []
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data)
    
    
    if(epoch % 10 is 0):
        val_loss = plot_and_loss(model, val_data,epoch)
        predict_future(model, val_data,20)
    else:
        val_loss = evaluate(model, val_data)
        
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)
    loss_list.append(val_loss)

    #if val_loss < best_val_loss:
    #    best_val_loss = val_loss
    #    best_model = model

    scheduler.step() 





def transform_stock_data(input_stock_price):
    #this function is for transforming stock data 
    #we first take log of the data and use minmaxscaler to make the max and the min of the price
    #fit between 0 and 1
    log_price = lognormal_from_price(np.array(input_stock_price))
    scaler = MinMaxScaler(feature_range=(-1, 1)) 
    scaled_price = scaler.fit_transform(log_price.reshape(-1, 1)).reshape(-1)
    return scaler,scaled_price

def inverse_transform_predict_data(scaler,output_sequence):
    transform_back = scaler.inverse_transform(output_sequence.reshape(-1, 1)).reshape(-1)
    exp_price = price_from_lognormal(transform_back)
    return exp_price


pred_sequence = inverse_transform_predict_data(scaler,np.array(pred_sequence))

pred_sequence[-5:].tofile('result/VZ.csv', sep = ',')#save your prediction result here

