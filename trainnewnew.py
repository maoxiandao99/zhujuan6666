import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
import pickle
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import matplotlib
import time
from data import sample_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyDataset(Dataset):
    def __init__(self, file_name, max_sample=-1):
        super(MyDataset, self).__init__()
        print('load data', file_name)
        self.param_dic = {}
        with open(file_name, "rb") as fp:
            data_dic, param_dic = pickle.load(fp)
        X = data_dic['X']
        X = [torch.from_numpy(x) for x in X]
        Y = data_dic['Y']
        H = data_dic['H']
        self.param_dic = param_dic

        if max_sample > 0:  # For debugging purpose
            print('*' * 50)
            print('max_sample =', max_sample)
            print('*' * 50)
            X = X[:max_sample]
            Y = Y[:max_sample]
            H = H[:max_sample]
        self.lengths = [_.shape[0] for _ in X]
        self.max_lengths = max(self.lengths)

        self.X = torch.stack([nn.ZeroPad2d((0, 0, 0, self.max_lengths - _.shape[0]))(_) for _ in X])
        self.Y = torch.from_numpy(np.array(Y))
        self.H = torch.from_numpy(np.array(H).reshape((-1, 1)))
        self.Z = torch.stack([nn.ZeroPad2d((0, 0, 0, self.max_lengths - i))(torch.ones((i, 1))) for i in self.lengths])
        self.lengths = torch.from_numpy(np.reshape(np.array(self.lengths), (-1, 1))).float()

        print('=' * 50)
        print('size X', self.X.shape)
        print('size Y', self.Y.shape)
        print('size H', self.H.shape)
        print('size Z', self.Z.shape)
        print('=' * 50)

    def get_dim(self):
        return self.X.shape[1], self.Y.shape[1]

    def __getitem__(self, index):
        x = self.X[index, :]  # The trajectories
        y = self.Y[index, :]  # The parameters
        l = self.lengths[index, :]  # Lengths of trajectories
        h = self.H[index, :]  # Spanning times
        z = self.Z[index, :]  # The ZERO masks of the trajectories for varying lengths
        return x, y, l, h, z

    def __len__(self):
        return len(self.X)

class loss_l1(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.eps = torch.tensor(1e-8).to(device)
        self.loss = nn.L1Loss()
        self.weight = weight

    def forward(self, x, y):
        M = x.shape[1]
        loss = []
        l_sum = 0
        for i in range(M):
            l = torch.mean(self.loss(x[:, i], y[:, i]))
            l_sum = l_sum + l * self.weight[i]
            loss.append(l)
        loss.insert(0, l_sum)
        return loss

class PENN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class, activation=''):
        super(PENN, self).__init__()
        self.type = 'LSTM'
        if activation == '':
            self.activation = 'leakyrelu'
        else:
            self.activation = activation
        self.init_param = [in_dim, hidden_dim, n_layer, n_class]
        print('init_param', self.init_param)
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.nn_size = 20

        self.linears = nn.ModuleList()
        for k in range(3):
            if k == 0:
                self.linears.append(nn.Linear(hidden_dim + 1, self.nn_size))
                self.linears.append(nn.BatchNorm1d(self.nn_size))
            else:
                self.linears.append(nn.Linear(self.nn_size, self.nn_size))
            if self.activation == 'elu':
                self.linears.append(torch.nn.ELU())
            elif self.activation == 'leakyrelu':
                self.linears.append(nn.LeakyReLU())
            elif self.activation == 'tanh':
                self.linears.append(nn.Tanh())
        self.linears.append(nn.Linear(self.nn_size, n_class))
        self.ones = torch.ones((1, self.hidden_dim)).to(device)
        self.init_weights()

    def _init_lstm(self, weight):
        for w in weight.chunk(4, 0):
            init.orthogonal_(w)

    def init_weights(self):
        self._init_lstm(self.lstm.weight_ih_l0)
        self._init_lstm(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

    def forward(self, x, l, h, z):
        _out, _ = self.lstm(x)
        out = _out * z
        out = torch.sum(out, dim=1)
        out = torch.div(out, torch.mm(l, self.ones))
        out = torch.cat([out, h], dim=1)
        for m in self.linears:
            out = m(out)
        return out

def train_net(config):
    P = config.param
    print('train file:', P['train_file'])
    print('eval file:', P['eval_file'])
    print('cuda availability', torch.cuda.is_available())

    train_data = MyDataset(P['train_file'])
    eval_data = MyDataset(P['eval_file'])

    num_epochs = 1
    init_weight_file = P['init_weight_file']
    param_name = P['param_name']
    model = PENN(train_data.X.shape[2],
                 P['lstm_fea_dim'],
                 P['lstm_layers'],
                 train_data.get_dim()[1],
                 activation=P['activation']).to(device)
    model = model.double()
    if not init_weight_file:
        model.init_weights()
    else:
        print('=' * 50)
        print('load model', init_weight_file)
        print('=' * 50)
        model_CKPT = torch.load(init_weight_file)
        model.load_state_dict(model_CKPT['state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), lr=P['learning_rate'])
    if init_weight_file:
        optimizer.load_state_dict(model_CKPT['optimizer'])
        init_epoch = model_CKPT['epoch']
    else:
        init_epoch = 0
    criterion = loss_l1(P['loss_weight'])
    train_loader = DataLoader(dataset=train_data, batch_size=P['batch_size'], shuffle=True, num_workers=1, drop_last=P['drop_last'])
    eval_loader = DataLoader(dataset=eval_data, batch_size=len(eval_data), shuffle=False, num_workers=1)

    save_path = os.path.join(config.save_path, P['architecture_name'])
    if not os.path.exists(os.path.join(save_path, 'runs')):
        os.makedirs(os.path.join(save_path, 'runs'))
    writer = SummaryWriter(os.path.join(save_path, 'runs'))

    for i, (x_eval, y_eval, l_eval, h_eval, z_eval) in enumerate(eval_loader):
        print(x_eval.shape)
        x_eval = x_eval.to(device)
        y_eval = y_eval.to(device)
        l_eval = l_eval.to(device)
        h_eval = h_eval.to(device)
        z_eval = z_eval.to(device)
    for epoch in range(init_epoch, num_epochs):
        start_time = time.time()
        for i, (x, y, l, h, z) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            l = l.to(device)
            h = h.to(device)
            z = z.to(device)
            outputs = model(x, l, h, z)
            losses = criterion(outputs, y)
            optimizer.zero_grad()
            losses[0].backward()
            optimizer.step()
        used_time = time.time() - start_time
        if (epoch + 1) % 1 == 0:
            with torch.no_grad():
                outputs = model(x_eval, l_eval, h_eval, z_eval)
                eval_errors = criterion(outputs, y_eval)
            writer.add_scalar('Loss/train', losses[0].item(), epoch + 1)
            writer.add_scalar('Loss/eval', eval_errors[0].item(), epoch + 1)
            param_num = len(param_name)
            for rr in range(param_num):
                writer.add_scalar('Loss/train_' + param_name[rr], losses[rr+1].item(), epoch + 1)
                writer.add_scalar('Loss/eval_' + param_name[rr], eval_errors[rr+1].item(), epoch + 1)
            writer.add_scalar('time/train', used_time, epoch + 1)
            print('Epoch [{}/{}], Loss: {:.4f}/{:.4f}, time: {:.1f}'
                  .format(epoch + 1, num_epochs, losses[0].item(), eval_errors[0], used_time))
            for param_group in optimizer.param_groups:
                print('rate', param_group['lr'])
            if eval_errors[0] == np.nan:
                print('train fail: nan')
                return
        if (epoch + 1) % 1 == 0:
            save_dict = {
                'network': model.type,
                'init_param': model.init_param,
                'activation': model.activation,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'param_dic': P
            }
            if not os.path.exists(os.path.join(save_path, 'model')):
                os.makedirs(os.path.join(save_path, 'model'))
            torch.save(save_dict, os.path.join(save_path, 'model/model_%05d.ckpt' % (epoch + 1)))

def set_title(ax, string, r=0.27, fs=17):
    min_y, max_y = ax.get_ylim()
    y = min_y - (max_y - min_y) * r
    min_x, max_x = ax.get_xlim()
    x = (min_x + max_x) * 0.5
    ax.text(x, y, string, horizontalalignment='center', verticalalignment='center', fontsize=fs)



def load_data(file_name):
    with open(file_name, "rb") as fp:
        data = pickle.load(fp)
    return data

def save_data(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)

def draw_gt_vs_estimated(config, saved_name='./data/temp.pkl'):
    param_name = config.param['param_name_latex']
    y, o = load_data(saved_name)
    alphabet = ['a', 'b', 'c', 'd']
    title = ['(' + alphabet[i] + ') ' + param_name[i].replace('$', r'\$') for i in range(len(param_name))]
    fig, ax = plt.subplots(1, len(title), figsize=(len(title) * 3.3, 3.4))
    for k in range(len(title)):
        ax[k].scatter(y[:, k], o[:, k], color='r', s=0.5)
        minv = min(min(y[:, k]), min(o[:, k]))
        maxv = max(max(y[:, k]), max(o[:, k]))
        ax[k].plot([minv, maxv], [minv, maxv], 'b')
        ax[k].axis('equal')
        ax[k].grid(True, which='minor')
        ax[k].set_xlabel('True values', fontsize=15)
        ax[k].set_ylabel('Estimated values', fontsize=15)
        set_title(ax[k], title[k], 0.34)
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, bottom=0.23, right=0.95, top=0.99, hspace=0.1, wspace=0.3)
    plt.show()
    plt.cla()


def evaluate_model(model, test_loader, criterion):
    model.eval()
    y_list = []
    o_list = []
    with torch.no_grad():
        for x, y, l, h, z in test_loader:
            x = x.to(device)
            y = y.to(device)
            l = l.to(device)
            h = h.to(device)
            z = z.to(device)
            outputs = model(x, l, h, z)
            y_list.append(y.cpu().numpy())
            o_list.append(outputs.cpu().numpy())
    y_np = np.concatenate(y_list, axis=0)
    o_np = np.concatenate(o_list, axis=0)
    save_data('./data/temp.pkl', (y_np, o_np))
    draw_gt_vs_estimated(config)

def train(config):
    print(config.param)
    train_net(config)
    
    # 使用与训练时相同的 config.param
    test_data_path = "/content/drive/MyDrive/Colab Notebooks/PENN-main/data/roughbergomi/roughbergomi_test.pkl"
    test_data = MyDataset(test_data_path)
    
    test_loader = DataLoader(dataset=test_data, batch_size=len(test_data), shuffle=False, num_workers=1)
    
    model = PENN(test_data.X.shape[2],
                 config.param['lstm_fea_dim'],
                 config.param['lstm_layers'],
                 test_data.get_dim()[1],
                 activation=config.param['activation']).to(device)
    model = model.double()
    
    # Load the model from the last checkpoint
    model_CKPT = torch.load(os.path.join(config.save_path, config.param['architecture_name'], 'model/model_%05d.ckpt' % 10))
    model.load_state_dict(model_CKPT['state_dict'])
    
    criterion = loss_l1(config.param['loss_weight'])
    evaluate_model(model, test_loader, criterion)



if __name__ == '__main__':
    from config_roughbergomi import Config, sampling_func

    config = Config()

    sample_data(config, sampling_func, mode='train')

    train(config)
