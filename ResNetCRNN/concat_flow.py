import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

os.chdir('./')

# save all train test results
A = np.load('./concat_training_losses.npy')
B = np.load('./concat_training_scores.npy')
C = np.load('./concat_test_loss.npy')
D = np.load('./concat_test_score.npy')

epochs = len(A)

# plot
fig = plt.figure(figsize=(16, 7))
plt.subplot(121)
plt.plot(np.arange(1, epochs + 1), A[:, -1])  # train loss (on epoch end)
plt.plot(np.arange(1, epochs + 1), C)         # test loss (on epoch end)
plt.title("model loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc="upper left")
# 2nd figure
plt.subplot(122)
plt.plot(np.arange(1, epochs + 1), B[:, -1])  # train loss (on epoch end)
plt.plot(np.arange(1, epochs + 1), D)         # test loss (on epoch end)
# plt.plot(histories.losses_val)
plt.title("CRNN scores")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train', 'test'], loc="upper left")
save_file = "./flow.png"
plt.savefig(save_file, dpi=600)
# plt.close(fig)
plt.show()


best_epoch = np.where(D==np.max(D))[0]
print('Best epoch: {}, validation accuracy: {:.2f}%'.format(best_epoch[0], 100 * np.max(D)))