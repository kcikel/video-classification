import numpy as np
import torch
from sklearn.metrics import accuracy_score
from pyswarm import pso

def reduce_classes(c):
    r = np.zeros(len(c))
    for i in range(len(c)):
        if c[i] == 0:
            r[i] = 0
        elif c[i] < 3:
            r[i] = 1
        elif c[i] < 4:
            r[i] = 2
        elif c[i] < 6:
            r[i] = 3
        elif c[i] < 8:
            r[i] = 4
        elif c[i] < 10:
            r[i] = 5
        elif c[i] >= 10:
            r[i] = 6
    return r

#def reduce_classes(c):
#    return c
    

def f(params):

    rgb_output = np.load('./rgb_output_otra.npy', allow_pickle=True)
    rgb_i = np.load('./rgb_i_otra.npy', allow_pickle=True)
    flow_output = np.load('./flow_output.npy', allow_pickle=True)
    flow_i = np.load('./flow_i.npy', allow_pickle=True)
    concat_output = np.load('./concat_output.npy', allow_pickle=True)
    concat_i = np.load('./concat_i.npy', allow_pickle=True)
    test_label = np.load('./test_label.npy', allow_pickle=True)
    
    a, b = params
    y_pred = []
    
    y = reduce_classes(test_label)
    
    
    r_out = []
    f_out = []
    c_out = []
    
    
    
    for i in range(len(rgb_i)):
        ri = np.where(rgb_i == i)
        fi = np.where(flow_i == i)
        ci = np.where(concat_i == i)
        r_out.append(rgb_output[ri])
        f_out.append(flow_output[fi])
        c_out.append(concat_output[ci])
    
    r_out = np.array(r_out)
    f_out = np.array(f_out)
    c_out = np.array(c_out)
    
    #a = 0.75
    #b = 0.15
    c = 1 - (a + b)
    output = (a*r_out + b*f_out + c*c_out)
    
    
    for o in output:
        x = o[0].cpu().data.squeeze().numpy()
        pred = np.where(x==max(x))
        y_pred.append(pred[0][0])
    
    y_pred = reduce_classes(y_pred)
    test_score = accuracy_score(y, y_pred)
    
    
    # show information
    print('\nTest set ({:d} samples):  Accuracy: {:.2f}%\n'.format(len(rgb_i), 100* test_score))
    return (-1 * test_score)
initial_guess = [0.75, 0.15]


lb = [0, 0]
ub = [1, 1]

xopt, fopt = pso(f, lb, ub)
print(xopt)
print(fopt)