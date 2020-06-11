
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torchvision.datasets.mnist as mnist_tv

import torch

import os


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            sorted(os.listdir(root))
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


class mnist():
    def __init__(self, flag='conv', is_tanh = False):
        datapath = 'dataset/mnist'
        self.X_dim = 784 # for mlp
        self.z_dim = 100
        self.y_dim = 10     # class num
        self.size = 28 # for conv
        self.channel = 1 # for conv
        #self.data = input_data.read_data_sets(datapath, one_hot=True)
        self.train_data = mnist_tv.MNIST(root='./dataset', train=True, download=True)
        self.val_data = mnist_tv.MNIST(root='./dataset', train=False, download=True)

        self.flag = flag
        self.is_tanh = is_tanh

    def load_target_feature_data(self, train_size, val_size, test_size, target_class = 10, mode = "conv", seed= 25) :

        train_data_list = list();
        train_label_list = list();
        val_data_list = list();
        val_label_list = list();
        test_data_list = list();
        test_label_list = list();

        np.random.seed(seed);

        torch.manual_seed(seed);


        for c in range(10) :
            if c != target_class :
                continue;

            label_info = self.train_data.targets.numpy();
            idx_c = np.where(label_info==c)[0];
            #np.random.shuffle(idx_c);
            idx_c = np.random.permutation(idx_c);
            train_data_list.append(self.train_data.data[idx_c[:train_size]].numpy()/255.0);
            t_label = np.zeros(shape=(train_size, 10));
            t_label[:, c] = 1.0;
            train_label_list.append(t_label);

            #train_data_list.append(self.data.train.images[:train_size]);
            #train_label_list.append(self.data.train.labels[:train_size]);

            label_info = self.val_data.targets;
            idx_c = np.where(label_info == c)[0];
            np.random.shuffle(idx_c);

            val_data_list.append(self.val_data.data[idx_c[:val_size]].numpy()/255.0);
            t_label = np.zeros(shape=(idx_c[:val_size].shape[0], 10));
            t_label[:, c] = 1.0;
            val_label_list.append(t_label);

        total_val_size = self.val_data.targets.shape[0];

        test_idx = np.arange(total_val_size);
        np.random.shuffle(test_idx);
        test_idx = test_idx[:test_size];

        test_dat = self.val_data.test_data[test_idx].numpy()/255.0;
        test_labels= self.val_data.test_labels.numpy();
        test_lab = np.eye(10)[test_labels];

        train_dat = np.vstack(train_data_list);
        train_label = np.vstack(train_label_list);

        val_dat = np.vstack(val_data_list);
        val_label = np.vstack(val_label_list);

        if mode == 'conv' :
            train_dat = np.reshape(train_dat, (train_dat.shape[0], self.channel, self.size, self.size));
            val_dat = np.reshape(val_dat, (val_dat.shape[0], self.channel, self.size, self.size, ));
            test_dat = np.reshape(test_dat, (test_dat.shape[0], self.channel , self.size, self.size));

        return train_dat, train_label,  val_dat, val_label, test_dat, test_lab, train_dat, train_label_list, val_dat, val_label_list;

    def load_feature_data(self, train_size, val_size, test_size, class_num = 10, mode = "fc", seed= 1) :

        tot_class_num = class_num;
        train_data_list = list();
        train_label_list = list();
        val_data_list = list();
        val_label_list = list();
        test_data_list = list();
        test_label_list = list();

        np.random.seed(seed);
        torch.manual_seed(seed);

        for c in range(tot_class_num) :

            label_info = self.train_data.targets.numpy();
            idx_c = np.where(label_info==c)[0];
            #np.random.shuffle(idx_c);
            idx_c = np.random.permutation(idx_c);
            train_data_list.append(self.train_data.data[idx_c[:train_size]].unsqueeze(1).numpy()/255.0);
            t_label = np.zeros(shape=(train_size, 10));
            t_label[:, c] = 1.0;
            train_label_list.append(t_label);

            #train_data_list.append(self.data.train.images[:train_size]);
            #train_label_list.append(self.data.train.labels[:train_size]);

            label_info = self.val_data.targets;
            idx_c = np.where(label_info == c)[0];
            np.random.shuffle(idx_c);

            val_data_list.append(self.val_data.data[idx_c[:val_size]].unsqueeze(1).numpy()/255.0);
            t_label = np.zeros(shape=(idx_c[:val_size].shape[0], 10));
            t_label[:, c] = 1.0;
            val_label_list.append(t_label);

        total_val_size = self.val_data.targets.shape[0];

        test_idx = np.arange(total_val_size);
        np.random.shuffle(test_idx);
        test_idx = test_idx[:test_size];

        test_dat = self.val_data.test_data[test_idx].unsqueeze(1).numpy()/255.0;
        test_labels= self.val_data.test_labels[test_idx].numpy();
        test_lab = np.eye(10)[test_labels];

        train_dat = np.vstack(train_data_list);
        train_label = np.vstack(train_label_list);

        val_dat = np.vstack(val_data_list);
        val_label = np.vstack(val_label_list);

        if mode == 'conv' :
            train_dat = np.reshape(train_dat, (train_dat.shape[0], self.channel, self.size, self.size));
            val_dat = np.reshape(val_dat, (val_dat.shape[0], self.channel, self.size, self.size, ));
            test_dat = np.reshape(test_dat, (test_dat.shape[0], self.channel , self.size, self.size));

        return train_dat, train_label,  val_dat, val_label, test_dat, test_lab, train_data_list, train_label_list, val_data_list, val_label_list;


    def test_mnist(self, batch_size=-1):
        if batch_size != -1 :
            batch_imgs = self.data.test._images[0:batch_size,:,:,:];
            batch_labels = self.data.test._labels[0:batch_size,:];
        else :
            batch_imgs = self.data.test._images;
            batch_labels = self.data.test._labels;

            batch_imgs = np.reshape(batch_imgs, (batch_imgs.shape[0], self.size, self.size, self.channel))

            return batch_imgs, batch_labels

    def data2fig(self, samples):
        if self.is_tanh:
            samples = (samples + 1)/2
            fig = plt.figure(figsize=(4, 4))
            gs = gridspec.GridSpec(4, 4)
            gs.update(wspace=0.05, hspace=0.05)

            for i, sample in enumerate(samples):
                ax = plt.subplot(gs[i])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(sample.reshape(self.size,self.size), cmap='Greys_r')
                return fig

