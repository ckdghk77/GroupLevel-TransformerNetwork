
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import dataset
from torch.utils.data import Sampler
import torch.nn.functional as F
from torch.autograd import Variable

from datas import *
from sklearn.datasets import make_moons
from sklearn.datasets import make_swiss_roll

import matplotlib.patches as patches
from matplotlib.patches import ArrowStyle
import matplotlib.pyplot as plt
import matplotlib


from torchvision import transforms
from os.path import join
import itertools
import torchvision
import torchvision.datasets.mnist as Mnist
import torchvision.datasets.cifar as cifar

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


import imgaug.augmenters as iaa
from scipy import ndimage, misc
from skimage.transform import resize
from skimage.transform import rotate

def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input)
    return soft_max_1d.transpose(axis, 0)


class CustomDataSet(data.Dataset) :

    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data[0];
        self.targets = data[1];


    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]


        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


class PermuteSampler(Sampler):
    r""" permute the samples  and randomly iter.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source , n_sequence = 5, batch_size =10):
        self.data_source = data_source
        self.n_sequence = n_sequence;

    def __iter__(self):
        n = len(self.data_source)

        return iter(torch.randperm(n)[:self.n_sequence].tolist())

    def __len__(self):
        return len(self.data_source)

class BatchPermuteSampler(Sampler) :
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    '''
    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch
    '''

    def __iter__(self):
        batch = []
        for b in range(self.batch_size) :
            b_list = [];
            for idx in self.sampler:
                b_list.append(idx)

            batch.append(b_list);

        yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


def load_data_toy_mnist_sequence(batch_size, train_size = 8, val_size = 10, test_size=10, sequence_num= 4, target_class=1, seed = 25, mode= 'conv', c_mode='move') :
    data = mnist();

    train_x, train_y, val_x, val_y, test_x, test_y, train_data_list, train_label_list, val_data_list, val_label_list = data.load_target_feature_data(
        train_size, val_size, test_size, target_class, mode= mode, seed=seed);

    src_dat = train_x[0];

    toy_train_list = list();

    toy_train_y_list = list();

    temp_source_val = np.linspace(0,5000, 784).astype(np.int32);
    temp_source_val = np.reshape(temp_source_val, [28, 28]);

    temp_list = list();

    if c_mode == 'move' :

        for i in range(train_size) :
            if mode == 'fc' :
                #src_dat_new = ndimage.rotate(np.squeeze(src_dat.reshape([28,28])), 50 * (i + 1), reshape=False)
                src_dat_new = ndimage.shift(np.squeeze(src_dat.reshape([28,28])), 2 * (i-2));
                toy_train_list.append(src_dat_new.reshape([784]));
                label_i = np.zeros(train_size, );
                label_i[i] = 1.0;
                toy_train_y_list.append(label_i)
            else :
                #src_dat_new = ndimage.rotate(np.squeeze(src_dat), 50*(i+1), reshape= False);
                src_dat_new = ndimage.shift(np.squeeze(src_dat), -2 * (i-2));

                temp_transform = ndimage.shift(np.squeeze(temp_source_val), 2 * (i));

                toy_train_list.append(np.expand_dims(src_dat_new,0));
                temp_list.append(np.expand_dims(temp_transform,0));


                label_i = np.zeros(train_size, );
                label_i[i] = 1.0;
                toy_train_y_list.append(label_i)

    elif c_mode == 'rot' :
        for i in range(train_size) :
            if mode == 'fc' :
                src_dat_new = ndimage.rotate(np.squeeze(src_dat.reshape([28,28])), 20 * (i-2), reshape=False)

                toy_train_list.append(src_dat_new.reshape([784]));
                label_i = np.zeros(train_size, );
                label_i[i] = 1.0;
                toy_train_y_list.append(label_i)
            else :
                src_dat_new = ndimage.rotate(np.squeeze(src_dat), 20*(i-2), reshape= False);

                temp_transform = ndimage.rotate(np.squeeze(temp_source_val), 20*(i), reshape= False);

                toy_train_list.append(np.expand_dims(src_dat_new, 0));
                temp_list.append(np.expand_dims(temp_transform, 0));

                label_i = np.zeros(train_size, );
                label_i[i] = 1.0;
                toy_train_y_list.append(label_i)


    train_x = np.stack(toy_train_list);
    train_y = np.stack(toy_train_y_list);
    train_data_list = np.stack(toy_train_list);


    train_x = torch.FloatTensor(train_x)
    val_x = torch.FloatTensor(val_x)

    label_train = torch.FloatTensor(np.argmax(train_y,1))
    label_val = torch.FloatTensor(np.argmax(val_y,1))


    train_data = TensorDataset(train_x, label_train)
    valid_data = TensorDataset(val_x, label_val)

    train_sampler = PermuteSampler(train_data, n_sequence=sequence_num);
    valid_sampler = PermuteSampler(valid_data, n_sequence=sequence_num);

    train_batch_sampler = BatchPermuteSampler(train_sampler, batch_size=batch_size, drop_last=False);
    valid_batch_sampler = BatchPermuteSampler(valid_sampler, batch_size=batch_size, drop_last=False);

    train_data_loader = DataLoader(train_data, batch_sampler=train_batch_sampler)
    valid_data_loader = DataLoader(valid_data, batch_sampler=valid_batch_sampler)


    return train_data_loader, valid_data_loader, train_data_list, train_label_list, val_data_list, val_label_list;


def load_data_mnist_sequence(batch_size, train_size = 8, val_size = 10, test_size=10, sequence_num= 4, target_class=1, seed = 25, mode= 'conv') :
    data = mnist();

    train_x, train_y, val_x, val_y, test_x, test_y, train_data_list, train_label_list, val_data_list, val_label_list = data.load_target_feature_data(
        train_size, val_size, test_size, target_class, mode= mode, seed=seed);


    train_x = torch.FloatTensor(train_x)
    val_x = torch.FloatTensor(val_x)

    label_train = torch.FloatTensor(train_y)
    label_val = torch.FloatTensor(np.stack(val_y))


    train_data = TensorDataset(train_x, label_train)
    valid_data = TensorDataset(val_x, label_val)

    train_sampler = PermuteSampler(train_data, n_sequence=sequence_num);
    valid_sampler = PermuteSampler(valid_data, n_sequence=sequence_num);

    train_batch_sampler = BatchPermuteSampler(train_sampler, batch_size=batch_size, drop_last=False);
    valid_batch_sampler = BatchPermuteSampler(valid_sampler, batch_size=batch_size, drop_last=False);

    train_data_loader = DataLoader(train_data, batch_sampler= train_batch_sampler)
    valid_data_loader = DataLoader(valid_data, batch_sampler=valid_batch_sampler)




    return train_data_loader, valid_data_loader, train_data_list, train_label_list, val_data_list, val_label_list;


def load_data_fmnist_sequence(batch_size, train_size = 8, val_size = 10, test_size=10, sequence_num= 4, target_class=1, seed = 25, mode= 'conv') :

    trainset = Mnist.FashionMNIST(root='./dataset', download=True, train=True);
    valset = Mnist.FashionMNIST(root='./dataset', download=True, train=False);

    np.random.seed(seed)
    torch.manual_seed(seed)


    tot_train_data = trainset.train_data
    tot_train_label = trainset.targets

    train_label_idx = (tot_train_label == target_class).nonzero();
    random_idx = torch.randperm(train_label_idx.shape[0])
    train_label_idx = train_label_idx[random_idx[:train_size]].squeeze();

    train_x = tot_train_data[train_label_idx].unsqueeze(1).float()/255.0;
    train_y = tot_train_label[train_label_idx];

    tot_val_data = valset.test_data
    tot_val_label = valset.targets;

    val_label_idx = (tot_val_label == target_class).nonzero();
    random_idx = torch.randperm(val_label_idx.shape[0])
    val_label_idx = val_label_idx[random_idx[:val_size]].squeeze();

    val_x = tot_val_data[val_label_idx].unsqueeze(1).float()/255.0;
    val_y = tot_val_label[val_label_idx];

    train_data = TensorDataset(train_x, train_y)
    valid_data = TensorDataset(val_x, val_y)

    train_sampler = PermuteSampler(train_data, n_sequence=sequence_num);
    valid_sampler = PermuteSampler(valid_data, n_sequence=sequence_num);

    train_batch_sampler = BatchPermuteSampler(train_sampler, batch_size=batch_size, drop_last=False);
    valid_batch_sampler = BatchPermuteSampler(valid_sampler, batch_size=batch_size, drop_last=False);

    train_data_loader = DataLoader(train_data, batch_sampler= train_batch_sampler)
    valid_data_loader = DataLoader(valid_data, batch_sampler=valid_batch_sampler)


    return train_data_loader, valid_data_loader, train_x.cpu().numpy(), train_y.cpu().numpy(), val_x.cpu().numpy(), val_y.cpu().numpy();


def load_data_custom_sequence(batch_size, train_size = 8, sequence_num= 4, seed = 25, transparent = False, is_rgb=True) :
    dir_name = "./dataset/custom/";

    custom_files = os.listdir(dir_name);

    random.seed(seed);
    random.shuffle(custom_files);

    reshape_size_x = 64;
    reshape_size_y = 64;

    train_data_list = list();
    train_label_list = list();

    val_data_list = list();
    val_label_list = list();

    for file_idx in range(len(custom_files)):

        file_name = custom_files[file_idx];

        if transparent == True :
            image_np = np.array(Image.open(dir_name + file_name).convert('RGB').resize((reshape_size_x,reshape_size_y)))/255.0;
        else:

            image_np = np.array(Image.open(dir_name + file_name).resize((reshape_size_x, reshape_size_y))) / 255.0;

        if file_idx < train_size:
            train_data_list.append(image_np);
            train_label_list.append(1);  # dummy label

        if file_idx < train_size:  # we don't consider a validation data for this dataset
            val_data_list.append(image_np);
            val_label_list.append(1);  # dummy label

    if is_rgb :
        train_x = np.transpose(np.stack(train_data_list), [0, 3, 1, 2]);
        val_x = np.transpose(np.stack(val_data_list), [0, 3, 1, 2]);
    else :
        train_x = np.expand_dims(np.stack(train_data_list),1);
        val_x = np.expand_dims(np.stack(val_data_list),1);

    train_y = np.stack(train_label_list);
    val_y = np.stack(val_label_list);

    train_x = torch.FloatTensor(train_x)
    val_x = torch.FloatTensor(val_x)
    label_train = torch.LongTensor(train_y)
    label_val = torch.LongTensor(val_y)

    train_data = TensorDataset(train_x, label_train)  # label doesn't need figr is only for generation
    valid_data = TensorDataset(val_x, label_val)

    train_sampler = PermuteSampler(train_data, n_sequence=sequence_num);
    valid_sampler = PermuteSampler(valid_data, n_sequence=sequence_num);

    train_batch_sampler = BatchPermuteSampler(train_sampler, batch_size=batch_size, drop_last=False);
    valid_batch_sampler = BatchPermuteSampler(valid_sampler, batch_size=batch_size, drop_last=False);

    train_data_loader = DataLoader(train_data, batch_sampler=train_batch_sampler)
    valid_data_loader = DataLoader(valid_data, batch_sampler=valid_batch_sampler)

    return train_data_loader, valid_data_loader, np.transpose(np.stack(train_data_list), [0, 3, 1,
                                                                                          2]), train_label_list, val_data_list, val_label_list;


def load_data_cifar_sequence(batch_size, train_size = 8, val_size = 10, test_size=10, sequence_num= 4, target_class=1, seed = 25, mode= 'conv') :

    trainset = cifar.CIFAR10(root='./dataset', download=True, train=True);
    valset = cifar.CIFAR10(root='./dataset', download=True, train=False);

    np.random.seed(seed)
    torch.manual_seed(seed)

    tot_train_data = trainset.data
    tot_train_label = np.stack(trainset.targets)

    train_label_idx = (tot_train_label == target_class).nonzero()[0];
    random_idx = np.random.permutation(train_label_idx.shape[0])
    train_label_idx = train_label_idx[random_idx[:train_size]].squeeze();

    train_x = np.transpose(tot_train_data[train_label_idx]/255.0, [0,3,1,2,]);
    train_y = tot_train_label[train_label_idx];

    tot_val_data = valset.data
    tot_val_label = np.stack(valset.targets);

    val_label_idx = (tot_val_label == target_class).nonzero()[0];
    random_idx = torch.randperm(val_label_idx.shape[0])
    val_label_idx = val_label_idx[random_idx[:val_size]].squeeze();

    val_x = np.transpose(tot_val_data[val_label_idx]/255.0, [0,3,1,2,]);
    val_y = tot_val_label[val_label_idx];

    train_data = TensorDataset(torch.FloatTensor(train_x), torch.LongTensor(train_y))
    valid_data = TensorDataset(torch.FloatTensor(val_x), torch.LongTensor(val_y))

    train_sampler = PermuteSampler(train_data, n_sequence=sequence_num);
    valid_sampler = PermuteSampler(valid_data, n_sequence=sequence_num);

    train_batch_sampler = BatchPermuteSampler(train_sampler, batch_size=batch_size, drop_last=False);
    valid_batch_sampler = BatchPermuteSampler(valid_sampler, batch_size=batch_size, drop_last=False);

    train_data_loader = DataLoader(train_data, batch_sampler=train_batch_sampler)
    valid_data_loader = DataLoader(valid_data, batch_sampler=valid_batch_sampler)

    return train_data_loader, valid_data_loader, train_x, train_y, val_x, val_y;



def load_data_omniglot_sequence(batch_size, train_size = 8, val_size = 10, test_size=10, sequence_num= 4, target_class=1, seed = 25, mode= 'conv') :

    data = Omniglot(root='./dataset', download=True, background=False, transform=transforms.Compose([lambda x: x.resize((28, 28))]))

    tot_img_list = list();
    tot_label_list = list();

    for (img, label) in data :
        tot_img_list.append(img);
        tot_label_list.append(label);

    tot_img = np.stack(tot_img_list);
    tot_label = np.stack(tot_label_list);

    train_data_list = list();
    train_label_list = list();
    val_data_list = list();
    val_label_list = list();

    np.random.seed(seed);
    for c in target_class:

        idx_c = np.where(tot_label == c)[0];
        # np.random.shuffle(idx_c);
        idx_c = np.random.permutation(idx_c);
        img_list = list();
        img_val_list = list();
        for ii in range(train_size) :
            resized_img = 1 - np.squeeze(tot_img[idx_c[ii]])/255.0;
            img_list.append(np.expand_dims(resized_img,0));

        for ii in range(train_size, idx_c.shape[0]) :
            resized_img = 1 - np.squeeze(tot_img[idx_c[ii]])/255.0;
            img_val_list.append(np.expand_dims(resized_img,0));


        train_data_list.extend(img_list);
        train_label_list.extend(tot_label[idx_c[:train_size]]);

        val_data_list.extend(img_val_list);
        val_label_list.extend(tot_label[idx_c[train_size:]]);

    train_dat = np.stack(train_data_list);
    train_label = np.stack(train_label_list);

    val_dat = np.stack(val_data_list);
    val_label = np.stack(val_label_list);


    train_x = train_dat;
    val_x = val_dat;


    train_x = torch.FloatTensor(train_x)
    val_x = torch.FloatTensor(val_x)

    label_train = torch.FloatTensor(train_label)
    label_val = torch.FloatTensor(val_label)


    train_data = TensorDataset(train_x, label_train)
    valid_data = TensorDataset(val_x, label_val)

    train_sampler = PermuteSampler(train_data, n_sequence=sequence_num);
    valid_sampler = PermuteSampler(valid_data, n_sequence=sequence_num);

    train_batch_sampler = BatchPermuteSampler(train_sampler, batch_size=batch_size, drop_last=False);
    valid_batch_sampler = BatchPermuteSampler(valid_sampler, batch_size=batch_size, drop_last=False);

    train_data_loader = DataLoader(train_data, batch_sampler=train_batch_sampler)
    valid_data_loader = DataLoader(valid_data, batch_sampler=valid_batch_sampler)

    return train_data_loader, valid_data_loader, train_dat, train_label_list, val_data_list, val_label_list;


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}

    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot;



def encode_onehot_rel(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}

    zero_arr = np.zeros(len(classes))

    n_label = list();
    for i in range(len(classes)) :
        label_i = np.where(labels == i)[0];
        for j in range(label_i.shape[0]) :
            if i <= j :
                n_label.append(zero_arr)
            else :
                n_label.append(np.identity(len(classes))[i, :])

        #for j in range(i) :
        #    n_label.append(zero_arr);

        #if i% labels[i] == max_label :
    #labels_onehot = np.array(list(map(classes_dict.get, labels)),
    #                         dtype=np.int32)
    return np.stack(n_label)


def encode_onehot_sen(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}

    zero_arr = np.zeros(len(classes))
    max_label = np.max(labels);
    n_label = list();
    for i in range(len(classes)) :

        for j in range(i) :
            n_label.append(zero_arr);

        for j in range(i, max_label) :
            n_label.append(np.identity(len(classes))[j+1, :])


        #if i% labels[i] == max_label :
    #labels_onehot = np.array(list(map(classes_dict.get, labels)),
    #                         dtype=np.int32)
    return np.stack(n_label)


def get_triu_indices(num_nodes):
    """Linear triu (upper triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    triu_indices = (ones.triu() - eye).nonzero().t()
    triu_indices = triu_indices[0] * num_nodes + triu_indices[1]
    return triu_indices


def get_tril_indices(num_nodes):
    """Linear tril (lower triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    tril_indices = (ones.tril() - eye).nonzero().t()
    tril_indices = tril_indices[0] * num_nodes + tril_indices[1]
    return tril_indices


def get_offdiag_indices(num_nodes):
    """Linear off-diagonal indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    offdiag_indices = (ones - eye).nonzero().t()
    offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
    return offdiag_indices


def get_triu_offdiag_indices(num_nodes):
    """Linear triu (upper) indices w.r.t. vector of off-diagonal elements."""
    triu_idx = torch.zeros(num_nodes * num_nodes)
    triu_idx[get_triu_indices(num_nodes)] = 1.
    triu_idx = triu_idx[get_offdiag_indices(num_nodes)]
    return triu_idx.nonzero()


def get_tril_offdiag_indices(num_nodes):
    """Linear tril (lower) indices w.r.t. vector of off-diagonal elements."""
    tril_idx = torch.zeros(num_nodes * num_nodes)
    tril_idx[get_tril_indices(num_nodes)] = 1.
    tril_idx = tril_idx[get_offdiag_indices(num_nodes)]
    return tril_idx.nonzero()




def nll_gaussian(preds, target, variance, weighting = None, add_const=False):


    neg_log_p = (((preds - target) ** 2) / (2 * variance ** 2))

    return torch.mean(neg_log_p)


