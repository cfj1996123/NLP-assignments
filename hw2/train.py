import os
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
import numpy as np

from dataset import Dataset
from model import language_model


# model parameters
batch_size = 5
seq_len = 10
embed_dim = 200
# training parameters
num_epoch = 100
snapshot_interval = 1
eval_interval = 1
base_lr = 0.001
# path info
model_folder = "model"
checkpoint_path = "model/model_epoch100"
train_path = "ptb_lm_small/train"
eval_path = "ptb_lm_small/dev"
test_path = "ptb_lm_small/test"


def batchify(data, batch_size):
    nbatch = data.size(0) // batch_size
    # trim off data so that it can be divided by batch_size
    data = data.narrow(0, 0, batch_size*nbatch)
    data = data.view(batch_size, -1).t().contiguous() # size: (length per batch, batch_size)
    return data


def train(verbal=True):
    # check access to gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    dataset = Dataset(train_path, eval_path, test_path)
    train = batchify(dataset.train, batch_size)
    eval = batchify(dataset.eval, batch_size)

    train = train.to(device)
    eval = eval.to(device)

    # load model, optimizer and loss function
    os.makedirs(model_folder, exist_ok=True)
    net = language_model(num_words=dataset.num_words, embed_dim=embed_dim)
    net.train()
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=base_lr, weight_decay=0.0001)
    entropy_loss = nn.CrossEntropyLoss(reduction="mean")

    if checkpoint_path != "":
        ## restore from checkpoint
        ckpt = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        net.load_state_dict(ckpt['model_params'])
        optimizer.load_state_dict(ckpt['optim_params'])
        epoch = ckpt['epoch']
    else:
        epoch = 0

    total_loss = []
    eval_loss = []
    # code your training process here
    while epoch < num_epoch:
        loss_in_epoch = []
        iterator = range(0, train.size(0)-seq_len, seq_len)
        if verbal:
            iterator = tqdm(iterator)
        for start in iterator:
            # zero the accumulated gradients
            optimizer.zero_grad()

            input = train[start:(start+seq_len), :]        #size: (seq_len, batch_size)
            truth = train[(start+1):(start+seq_len+1), :]  #size: (seq_len, batch_size)
            output = net.forward(input)                    #size: (seq_len, batch_size, num_words)

            # compute training loss
            output = output.permute(0, 2, 1)               #size: (seq_len, num_words, batch_size)
            loss = entropy_loss(output, truth)

            # do gradient descent
            loss.backward()
            nn.utils.clip_grad_norm(net.parameters(), 10.0)
            optimizer.step()
            loss_in_epoch.append(loss.item())

            if verbal:
                description = 'epoch:{}, iteration:{}, current loss:{}, ' \
                              'mean loss:{}'.format(epoch, start//seq_len, loss.item(), np.mean(loss_in_epoch))
                #baseline loss: -ln(1/num_words) = 8.37
                iterator.set_description(description)

        total_loss.append(np.mean(loss_in_epoch))
        epoch += 1

        # evaluation on eval set
        net.eval()
        eval_loss_in_epoch = []
        for start in range(0, eval.size(0) - seq_len, seq_len):
            input = eval[start:(start + seq_len), :]  # size: (seq_len, batch_size)
            truth = eval[(start + 1):(start + seq_len + 1), :]  # size: (seq_len, batch_size)
            output = net.forward(input)  # size: (seq_len, batch_size, num_words)

            # compute training loss
            output = output.permute(0, 2, 1)  # size: (seq_len, num_words, batch_size)
            loss = entropy_loss(output, truth)
            eval_loss_in_epoch.append(loss.item())

        eval_loss.append(np.mean(eval_loss_in_epoch))
        net.train()

        # write averaged loss in this epoch into a file
        with open("loss.txt", "a+") as f:
            f.write("epoch: {}, training loss: {}, eval loss:{}\n".\
                    format(epoch,round(np.mean(loss_in_epoch),3), round(np.mean(eval_loss_in_epoch),3)))

        print("epoch: {}, training loss: {}, eval loss:{}".\
                    format(epoch,round(np.mean(loss_in_epoch),3), round(np.mean(eval_loss_in_epoch),3)))
        # save model
        if epoch % snapshot_interval == 0 or epoch == num_epoch:
            ckpt_path = os.path.join(model_folder, 'model_epoch{}'.format(epoch))
            torch.save({'epoch': epoch,
                        'model_params': net.state_dict(),
                        'optim_params': optimizer.state_dict()},
                       ckpt_path
                       )



def test():
    # check access to gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    dataset = Dataset(train_path, eval_path, test_path)
    test = batchify(dataset.test, batch_size)
    test = test.to(device)

    net = language_model(num_words=dataset.num_words, embed_dim=embed_dim)
    net.eval()
    net = net.to(device)
    entropy_loss = nn.CrossEntropyLoss(reduction="mean")

    ## restore from checkpoint
    assert checkpoint_path!="", "invalid checkpoint!"
    ckpt = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    net.load_state_dict(ckpt['model_params'])
    epoch = ckpt['epoch']

    test_loss = []
    for start in range(0, test.size(0) - seq_len, seq_len):
        input = test[start:(start + seq_len), :]  # size: (seq_len, batch_size)
        truth = test[(start + 1):(start + seq_len + 1), :]  # size: (seq_len, batch_size)
        output = net.forward(input)  # size: (seq_len, batch_size, num_words)

        # compute training loss
        output = output.permute(0, 2, 1)  # size: (seq_len, num_words, batch_size)
        loss = entropy_loss(output, truth)
        test_loss.append(loss.item())

    print("test loss: {}".format(np.mean(test_loss)))



if __name__ == "__main__":
    # train(verbal=False)
    test()













