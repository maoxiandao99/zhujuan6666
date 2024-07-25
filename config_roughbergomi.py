import os
import numpy as np
import pickle
import time
from data import generate_param

class Config:
    def __init__(self):
        # The path:
        # data_root/system_name
        # data_root/system_name/model -- for saving model weights
        # data_root/system_name/runs -- for saving tensorboard logs

        self.data_root = './data'
        self.system_name = 'roughbergomi'
        self.save_path = os.path.join(self.data_root, self.system_name)

        param_dic = {
            #####################################
            # Parameters for sampling
            #####################################

            # Variable parameter ranges of the system
            'hurst_index': [0, 1],
            'initial_volatility': [0, 1],
            'vol_of_vol': [0, 5],
            'correlation': [-1, 0],
            'param_name': ['hurst_index', 'initial_volatility', 'vol_of_vol', 'correlation'],
            'param_name_latex': ['$H$', '$\sigma_0$', '$\eta$', '$\rho$'],

            # The lengths of trajectories (integers)
            'N': [100, 400],

            # the time spans
            'T': [0.5, 1],

            # Numbers and file names of trajectories for training, evaluation and test, respectively.
            'train_num': 5000,
            'eval_num': 500,
            'test_num': 2000,

            'train_file': os.path.join(self.save_path, 'roughbergomi_train.pkl'),
            'eval_file': os.path.join(self.save_path, 'roughbergomi_eval.pkl'),
            'test_file': os.path.join(self.save_path, 'roughbergomi_test.pkl'),

            #####################################
            # Parameters for training
            #####################################

            'epochs': 50,
            'batch_size': 1600,
            'learning_rate': 0.001,
             # The architecture of the PENN
            'lstm_layers': 4,
            'lstm_fea_dim': 25,  # The dimension of the LSTM features
            'activation': 'elu',  # A slightly better than ReLU and atan
            'drop_last': True,  # Throw away the last mini batch in each epoch

            'architecture_name': 'roughbergomi_model_1',

            #####################################
            # Parameters for testing
            #####################################
            'eval_model_file': '',
            'test_model_file': '',

            # Weights of loss function for every parameter.
            'loss_weight': [1, 1, 1, 1],
        }
        self.param = param_dic

def fbm(H, N, L):
    """ Generate a fractional Brownian motion path with Hurst index H. """
    B = np.zeros(N)
    T = L / N
    for n in range(1, N):
        B[n] = B[n - 1] + np.random.normal(0, T**H)
    return B

def sampling_func(M=1000, hurst_index=[0, 1], initial_volatility=[0, 1], vol_of_vol=[0, 5], correlation=[-1, 0], N=[100, 400], T=[0.5, 1], save_name='roughbergomi.pkl'):
    xs = []
    ys = []
    NUM_FOR_SHOW = 2000
    N = generate_param(N, M)
    T = generate_param(T, M)
    dt = T / N
    hurst_index = generate_param(hurst_index, M)
    initial_volatility = generate_param(initial_volatility, M)
    vol_of_vol = generate_param(vol_of_vol, M)
    correlation = generate_param(correlation, M)
    
    time_start = time.time()
    for id in range(M):
        if (id+1) % NUM_FOR_SHOW == 0:
            time_end = time.time()
            time_used = time_end - time_start
            time_left = time_used / (id + 1) * (M - id)
            print(id+1, 'of', M, '%.1fs used %.1fs left' % (time_used, time_left))

        _N = int(N[id])
        x = trajectory_roughbergomi(_N, dt[id], hurst_index[id], initial_volatility[id], vol_of_vol[id], correlation[id])
        xs.append(np.array([x]).T)
        ys.append([hurst_index[id], initial_volatility[id], vol_of_vol[id], correlation[id]])

    with open(save_name, 'wb') as f:
        pickle.dump([xs, T[:len(xs)].tolist(), ys], f)

def trajectory_roughbergomi(N, dt, H, sigma_0, eta, rho, S0=1.0):
    """ Generate a Rough Bergomi model price trajectory. """
    fbm1 = fbm(H, N, 1.0)
    fbm2 = fbm(H, N, 1.0)
    
    volatility = np.zeros(N)
    volatility[0] = sigma_0

    for i in range(1, N):
        dW1 = fbm1[i] - fbm1[i - 1]
        dW2 = fbm2[i] - fbm2[i - 1]
        volatility[i] = volatility[i - 1] + eta * volatility[i - 1] * np.sqrt(dt) * (rho * dW1 + np.sqrt(1 - rho**2) * dW2)
    
    price = np.zeros(N)
    price[0] = S0
    integral_sigma_dW = 0
    integral_sigma_squared = 0

    for i in range(1, N):
        dW1 = fbm1[i] - fbm1[i - 1]
        integral_sigma_dW += volatility[i-1] * dW1
        integral_sigma_squared += volatility[i-1]**2 * dt
        price[i] = S0 * np.exp(integral_sigma_dW - 0.5 * integral_sigma_squared)
    
    return price
