import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from functions_test import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

import time


# set path
data_path = "/home/lsd/USTC/I3D_Finetune/data/handwash/"    # define UCF-101 RGB data path
flow_data_path = "/home/lsd/USTC/I3D_Finetune/data/handwash_flow_xy/"  
rgb_save_model_path1 = "/media/lsd/Disco 3/handwash_otra/"
rgb_save_model_path2 = "_epoch158.pth" #131
flow_save_model_path1 = "/media/lsd/Disco 3/flow_ckpt_otra/"
flow_save_model_path2 = "_epoch135.pth"
concat_save_model_path1 = "/media/lsd/Disco 3/flow_ckpt/"
concat_save_model_path2 = "_epoch252.pth"


# EncoderCNN architecture
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 512   # latent dim extracted by 2D CNN
res_size = 224        # ResNet image size
dropout_p = 0.5       # dropout probability

# DecoderRNN architecture
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256
# training parameters
k = 12             # number of target category
epochs = 180        # training epochs
batch_size = 80  
learning_rate = 1e-3
log_interval = 10   # interval for displaying training info

# Select which frame to begin & end in videos
begin_frame, end_frame, skip_frame = 1, 128, 1


def validation(model, device, optimizer, test_loader):
    # set model as testing mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    test_loss = 0
    outputs = []
    all_i = []
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, yi in test_loader:
            # distribute data to device

                
            X, yi = X.to(device), yi.to(device).view(-1, )
            y = []
            i = []

            for t in range(len(yi)):
                if t % 2 == 0:
                    y.append(yi[t])
                else:
                    i.append(yi[t])
            #print(y)
            #print(i)
            output = rnn_decoder(cnn_encoder(X))

            #loss = F.cross_entropy(output, y, reduction='sum')
            #test_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)
            outputs.extend(output)
            all_i.extend(i)


    #test_loss /= len(test_loader.dataset)

    # compute accuracy

    all_y_pred = torch.stack(all_y_pred, dim=0)
    all_y = torch.stack(all_y, dim=0)
    all_i = torch.stack(all_i, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))

    return test_loss, test_score, outputs, all_y, all_y_pred, all_i

def test(model, device, optimizer, test_loader):
    # set model as testing mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    test_loss = 0
    outputs = []
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )

            output = rnn_decoder(cnn_encoder(X))

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)
            outputs.extend(output)

    test_loss /= len(test_loader.dataset)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))

    return test_loss, test_score, outputs, all_y, all_y_pred


# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

# Data loading parameters
params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}


    
#action_names = ['1','2L','2R','3','4L','4R','5L','5R','6L','6R','7L','7R']

action_names = ['01','02','03','04','05','06','07','08','09','10','11','12']
# convert labels -> category
le = LabelEncoder()
le.fit(action_names)

# show how many classes there are
list(le.classes_)

# convert category -> 1-hot
action_category = le.transform(action_names).reshape(-1, 1)
enc = OneHotEncoder()
enc.fit(action_category)

# # example
# y = ['HorseRace', 'YoYo', 'WalkingWithDog']
# y_onehot = labels2onehot(enc, le, y)
# y2 = onehot2labels(le, y_onehot)

actions = []
fnames = os.listdir(data_path)

all_names = []
for f in fnames:
    loc1 = f.find('_A_')
    loc2 = f.rfind('_G')
    actions.append(f[(loc1 + 3): loc2])

    all_names.append(f)

#%%
# list all data files
all_X_list = all_names                  # all video file names
all_y_list = labels2cat(le, actions)    # all video labels

# train, test split
train_list, test_list, train_label, test_label = train_test_split(all_X_list, all_y_list, test_size=0.25, random_state=42)

transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

test_label_i = []
for i in range(len(test_label)):
    test_label_i.append([test_label[i], i])

valid_set = Dataset_CRNN(data_path, test_list, test_label_i, selected_frames, transform=transform)

#train_loader = data.DataLoader(train_set, **params)
valid_loader = data.DataLoader(valid_set, **params)


# Create model
cnn_encoder = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, 
                         h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)


# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    rgb_cnn_encoder = nn.DataParallel(cnn_encoder)
    rgb_rnn_decoder = nn.DataParallel(rnn_decoder)

    # Combine all EncoderCNN + DecoderRNN parameters
    rgb_crnn_params = list(rgb_cnn_encoder.module.fc1.parameters()) + list(rgb_cnn_encoder.module.bn1.parameters()) + \
                  list(rgb_cnn_encoder.module.fc2.parameters()) + list(rgb_cnn_encoder.module.bn2.parameters()) + \
                  list(rgb_cnn_encoder.module.fc3.parameters()) + list(rgb_rnn_decoder.parameters())

elif torch.cuda.device_count() == 1:
    print("Using", torch.cuda.device_count(), "GPU!")
    # Combine all EncoderCNN + DecoderRNN parameters
    rgb_crnn_params = list(rgb_cnn_encoder.fc1.parameters()) + list(rgb_cnn_encoder.bn1.parameters()) + \
                  list(rgb_cnn_encoder.fc2.parameters()) + list(rgb_cnn_encoder.bn2.parameters()) + \
                  list(rgb_cnn_encoder.fc3.parameters()) + list(rgb_rnn_decoder.parameters())
                  
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    flow_cnn_encoder = nn.DataParallel(cnn_encoder)
    flow_rnn_decoder = nn.DataParallel(rnn_decoder)

    # Combine all EncoderCNN + DecoderRNN parameters
    flow_crnn_params = list(flow_cnn_encoder.module.fc1.parameters()) + list(flow_cnn_encoder.module.bn1.parameters()) + \
                  list(flow_cnn_encoder.module.fc2.parameters()) + list(flow_cnn_encoder.module.bn2.parameters()) + \
                  list(flow_cnn_encoder.module.fc3.parameters()) + list(flow_rnn_decoder.parameters())

elif torch.cuda.device_count() == 1:
    print("Using", torch.cuda.device_count(), "GPU!")
    # Combine all EncoderCNN + DecoderRNN parameters
    flow_crnn_params = list(flow_cnn_encoder.fc1.parameters()) + list(flow_cnn_encoder.bn1.parameters()) + \
                  list(flow_cnn_encoder.fc2.parameters()) + list(flow_cnn_encoder.bn2.parameters()) + \
                  list(flow_cnn_encoder.fc3.parameters()) + list(flow_rnn_decoder.parameters())

# if torch.cuda.device_count() > 1:
#     print("Using", torch.cuda.device_count(), "GPUs!")
#     concat_cnn_encoder = nn.DataParallel(cnn_encoder)
#     concat_rnn_decoder = nn.DataParallel(rnn_decoder)

#     # Combine all EncoderCNN + DecoderRNN parameters
#     concat_crnn_params = list(concat_cnn_encoder.module.fc1.parameters()) + list(concat_cnn_encoder.module.bn1.parameters()) + \
#                   list(concat_cnn_encoder.module.fc2.parameters()) + list(concat_cnn_encoder.module.bn2.parameters()) + \
#                   list(concat_cnn_encoder.module.fc3.parameters()) + list(concat_rnn_decoder.parameters())

# elif torch.cuda.device_count() == 1:
#     print("Using", torch.cuda.device_count(), "GPU!")
#     # Combine all EncoderCNN + DecoderRNN parameters
#     concat_crnn_params = list(concat_cnn_encoder.fc1.parameters()) + list(concat_cnn_encoder.bn1.parameters()) + \
#                   list(concat_cnn_encoder.fc2.parameters()) + list(concat_cnn_encoder.bn2.parameters()) + \
#                   list(concat_cnn_encoder.fc3.parameters()) + list(concat_rnn_decoder.parameters())


rgb_optimizer = torch.optim.Adam(rgb_crnn_params, lr=learning_rate)
flow_optimizer = torch.optim.Adam(flow_crnn_params, lr=learning_rate)
#concat_optimizer = torch.optim.Adam(concat_crnn_params, lr=learning_rate)





i = 158
rgb_cnn_encoder.load_state_dict(torch.load(rgb_save_model_path1 + "cnn_encoder" + "_epoch" + str(i) + ".pth"))
rgb_rnn_decoder.load_state_dict(torch.load(rgb_save_model_path1 + "rnn_decoder" + "_epoch" + str(i) + ".pth"))
rgb_optimizer.load_state_dict(torch.load(rgb_save_model_path1 + "optimizer" + "_epoch" + str(i) + ".pth"))

rgb_test_loss, rgb_test_score, rgb_output, rgb_y, rgb_y_pred, rgb_i = validation([rgb_cnn_encoder, rgb_rnn_decoder], device, rgb_optimizer, valid_loader)

flow_cnn_encoder.load_state_dict(torch.load(flow_save_model_path1 + "cnn_encoder" + flow_save_model_path2))
flow_rnn_decoder.load_state_dict(torch.load(flow_save_model_path1 + "rnn_decoder" + flow_save_model_path2))
flow_optimizer.load_state_dict(torch.load(flow_save_model_path1 + "optimizer" + flow_save_model_path2))


flow_valid_set = Dataset_CRNN_flow(flow_data_path, test_list, test_label_i, selected_frames, transform=transform)

flow_valid_loader = data.DataLoader(flow_valid_set, **params)

flow_test_loss, flow_test_score, flow_output, flow_y, flow_y_pred, flow_i = validation([flow_cnn_encoder, flow_rnn_decoder], device, flow_optimizer, flow_valid_loader)


# concat_valid_set = Dataset_CRNN2(data_path, flow_data_path, test_list, test_label_i, selected_frames, transform=transform)

# concat_cnn_encoder.load_state_dict(torch.load(concat_save_model_path1 + "cnn_encoder" + concat_save_model_path2))
# concat_rnn_decoder.load_state_dict(torch.load(concat_save_model_path1 + "rnn_decoder" + concat_save_model_path2))
# concat_optimizer.load_state_dict(torch.load(concat_save_model_path1 + "optimizer" + concat_save_model_path2))

# concat_valid_loader = data.DataLoader(concat_valid_set, **params)

# concat_test_loss, concat_test_score, concat_output, concat_y, concat_y_pred, concat_i = validation([concat_cnn_encoder, concat_rnn_decoder], device, concat_optimizer, concat_valid_loader)


#%%

rgb_i = rgb_i.cpu().data.squeeze().numpy()
flow_i = flow_i.cpu().data.squeeze().numpy()

np.save('./rgb_output_otra.npy', rgb_output)
np.save('./flow_output_otra.npy', flow_output)
np.save('./rgb_i_otra.npy', rgb_i)
np.save('./flow_i_otra.npy', flow_i)
#np.save('./test_label.npy', test_label)
y_pred = []

y = test_label

r_output = torch.stack(rgb_output, dim=0)
f_output = torch.stack(flow_output, dim=0)

r_out = []
f_out = []

#rgb_i = rgb_i.cpu().data.squeeze().numpy()
#flow_i = flow_i.cpu().data.squeeze().numpy()

for i in range(len(rgb_i)):
    ri = np.where(rgb_i == i)
    fi = np.where(flow_i == i)
    r_out.append(r_output[ri])
    f_out.append(f_output[fi])

r_out = np.array(r_out)
f_out = np.array(f_out)

a = 0.90
b = 1 - a
output = (a*r_out + b*f_out)
output = np.array(output)



for o in output:
    x = o.cpu().data.squeeze().numpy()
    pred = np.where(x==max(x))
    y_pred.append(pred[0][0])


test_score = accuracy_score(y, y_pred)


# show information
print('\nTest set ({:d} samples):  Accuracy: {:.2f}%\n'.format(len(rgb_i), 100* test_score))
