import argparse
from tqdm import tqdm
import multiprocessing
from numpy.lib.format import open_memmap

parser = argparse.ArgumentParser(description='Dataset Preprocessing')
parser.add_argument('--use_mp', type=bool, default=False, help='use multi processing or not')
parser.add_argument('--modal', type=str, default='bone', help='use multi processing or not')

# uav graph
    # (10, 8), (8, 6), (9, 7), (7, 5), # arms
    # (15, 13), (13, 11), (16, 14), (14, 12), # legs
    # (11, 5), (12, 6), (11, 12), (5, 6), # torso
    # (5, 0), (6, 0), (1, 0), (2, 0), (3, 1), (4, 2) # nose, eyes and ears

sets = {'train', 'test'}
# sets = {'train', 'val'}

parts = {'joint', 'bone'}
graph = ((10, 8), (8, 6), (9, 7), (7, 5), (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6), (11, 12), (5, 6), (5, 0), (6, 0), (1, 0), (2, 0), (3, 1), (4, 2))

# bone
def gen_bone(set):
    print(set)
    data = open_memmap('./data/{}_joint.npy'.format(set),mode='r')
    N, C, T, V, M = data.shape
    fp_sp = open_memmap('./data/{}_bone.npy'.format(set),dtype='float32',mode='w+',shape=(N, 3, T, V, M))
    for v1, v2 in tqdm(graph):
        fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]

# jmb
def merge_joint_bone_data(set):
    print(set)
    data_jpt = open_memmap('./data/{}_joint.npy'.format(set), mode='r')
    data_bone = open_memmap('./data/{}_bone.npy'.format(set), mode='r')
    N, C, T, V, M = data_jpt.shape
    data_jpt_bone = open_memmap('./data/{}_joint_bone.npy'.format(set), dtype='float32', mode='w+', shape=(N, 6, T, V, M))
    data_jpt_bone[:, :C, :, :, :] = data_jpt
    data_jpt_bone[:, C:, :, :, :] = data_bone

# motion  
def gen_motion(set,part):
    print(set, part)
    data = open_memmap('./data/{}_{}.npy'.format(set, part),mode='r')
    N, C, T, V, M = data.shape
    fp_sp = open_memmap('./data/{}_{}_motion.npy'.format(set, part),dtype='float32',mode='w+',shape=(N, 3, T, V, M))
    for t in tqdm(range(T - 1)):
        fp_sp[:, :, t, :, :] = data[:, :, t + 1, :, :] - data[:, :, t, :, :]
    fp_sp[:, :, T - 1, :, :] = 0

if __name__ == '__main__':
    args = parser.parse_args()
    # Multiprocessing
    if args.use_mp:
        processes = []
        if args.modal == 'bone':   
            for set in sets:
                process = multiprocessing.Process(target=gen_bone, args=(set,))
                processes.append(process)
                process.start()
        elif args.modal == 'jmb':
            for set in sets:
                process = multiprocessing.Process(target=merge_joint_bone_data, args=(set,))
                processes.append(process)
                process.start()
        elif args.modal == 'motion':
            for set in sets:
                for part in parts:
                    process = multiprocessing.Process(target=gen_motion, args=(set, part))
                    processes.append(process)
                    process.start()
        else:
            raise ValueError('Invalid Modal')
        for process in processes:
            process.join()
    # Singleprocessing
    elif not args.use_mp:
        if args.modal == 'bone':   
            for set in sets:
                gen_bone(set)
        elif args.modal == 'jmb':
            for set in sets:
                merge_joint_bone_data(set)
        elif args.modal == 'motion':
            for set in sets:
                for part in parts:
                    gen_motion(set, part)
        else:
            raise ValueError('Invalid Modal')
    else:
        raise ValueError('Invalid use_mp set,only True or False')
