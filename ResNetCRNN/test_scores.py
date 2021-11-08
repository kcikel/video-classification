import numpy as np
import torch
from sklearn.metrics import accuracy_score

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
    
rgb_output = np.load('./rgb_output_otra.npy', allow_pickle=True)
rgb_i = np.load('./rgb_i_otra.npy', allow_pickle=True)
flow_output = np.load('./flow_output_otra.npy', allow_pickle=True)
flow_i = np.load('./flow_i_otra.npy', allow_pickle=True)
concat_output = np.load('./concat_output.npy', allow_pickle=True)
concat_i = np.load('./concat_i.npy', allow_pickle=True)
test_label = np.load('./test_label.npy', allow_pickle=True)


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

#a , b = [0.33333, 0.33333]
a , b = [0, 1]
#a , b = [0.608, 0.183]
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
#%%
from sklearn.metrics import classification_report
print(classification_report(y, y_pred))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y, y_pred)
FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
TP = np.diag(confusion_matrix)
TN = confusion_matrix.sum() - (FP + FN + TP)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)
rec = TP/(TP+FN)
prec = TP/(TP+FP)
f1 = 2*(prec*rec)/(prec+rec)

#%%
import matplotlib.pyplot as plt

#confusion_matrix = cf

fig, ax = plt.subplots()

min_val, max_val = 0, 15


dim=np.arange(1,13,1)

alpha = ["1","2","3","4","5","6","7","8","9","10","11","12"]

#dim=np.arange(1,8,1)

#alpha = ["1","2","3","4","5","6","7"]


#plt.xlim([0, 12])
#plt.ylim([12, 0])
ax.matshow(confusion_matrix, cmap=plt.cm.Blues)

for i in range(12):
    for j in range(12):
        c = confusion_matrix[j,i]
        ax.text(i, j, str(int(c)), va="center", ha="center")
        
#for (x, y), value in np.ndenumerate(confusion_matrix):
#    plt.text(x+1, y+1, value, va="center", ha="center")

#plt.xticks(dim)     
#plt.yticks(dim)

xaxis = np.arange(len(alpha))
ax.set_xticks(xaxis)
ax.set_yticks(xaxis)
ax.set_xticklabels(alpha)
ax.set_yticklabels(alpha)

     
#plt.grid()
plt.show()