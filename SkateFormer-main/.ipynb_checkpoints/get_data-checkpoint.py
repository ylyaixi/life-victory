import numpy as np

data = np.load(r'./data/train_joint.npy')
label = np.load(r'./data/train_label.npy')

for (i , data_numpy) in enumerate(data):
    if(np.all(data_numpy.sum(0).sum(-1).sum(-1) == 0)):
        print(i)

data = np.delete(data,13638,0)
label = np.delete(label,13638,0)

np.save(r'./data/train_joint_new',data)
np.save(r'./data/train_label_new',label)

