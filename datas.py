import os,sys
import scipy.misc
from glob import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torchvision.datasets import omniglot

import torch.utils.data as data
from PIL import Image, ImageFile
import imgaug.augmenters as iaa
import csv

#ImageFile.LOAD_TRUNCATED_IMAGES = True;
from torchvision import transforms
import torchvision.datasets.mnist as mnist_tv

import torch
from random import randint
import random


import csv

#from pytorch_pretrained_bert import BertModel, BertTokenizer
#from pytorch_pretrained_bert.modeling import BertForSequenceClassification

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

class Omniglot(data.Dataset):
    """`Omniglot <https://github.com/brendenlake/omniglot>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        background (bool, optional): If True, creates dataset from the "background" set, otherwise
            creates from the "evaluation" set. This terminology is defined by the authors.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset zip files from the internet and
            puts it in root directory. If the zip files are already downloaded, they are not
            downloaded again.
    """
    folder = 'omniglot-py'
    download_url_prefix = 'https://github.com/brendenlake/omniglot/raw/master/python'
    zips_md5 = {
        'images_background': '68d2efa1b9178cc56df9314c21c6e718',
        'images_evaluation': '6b91aef0f799c5bb55b94e3f2daec811'
    }

    def __init__(self, root, background=True,
                 transform=None, target_transform=None,
                 download=False):
        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.background = background
        self.transform = transform
        self.target_transform = target_transform


        self.target_folder = os.path.join(self.root, self._get_target_folder())
        self._alphabets = sorted(os.listdir(self.target_folder))
        self._characters = sum([[os.path.join(a, c) for c in sorted(os.listdir(os.path.join(self.target_folder, a)))]
                                for a in self._alphabets], [])
        self._character_images = [[(image, idx) for image in sorted(list_files(os.path.join(self.target_folder, character), '.png'))]
                                  for idx, character in enumerate(self._characters)]
        self._flat_character_images = sum(self._character_images, [])

    def __len__(self):
        return len(self._flat_character_images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target character class.
        """

        image_name, character_class = self._flat_character_images[index]
        image_path = os.path.join(self.target_folder, self._characters[character_class], image_name)
        image = Image.open(image_path, mode='r').convert('L')

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            character_class = self.target_transform(character_class)

        return image, character_class



    def _get_target_folder(self):
        return 'images_background' if self.background else 'images_evaluation'



class ISIC_data():
    def __init__(self, flag='conv', is_tanh=False):
        #self.X_dim = sample_data.shape[0]*sample_data.shape[1]  # for mlp
        self.y_dim = 2  # class num    #### it is usually a regressor
        self.size = 256  # for conv
        #self.size_r = sample_data.shape[0];
        #self.size_c = sample_data.shape[1];
        self.channel = 3  # for conv

        self.flag = flag
        self.is_tanh = is_tanh

    def data_from_file(self,data_file_path = 'data/ISIC-2017_Training_Data',  label_file_path = 'data/ISIC-2017_Label_Data/ISIC-2017_3.txt'):
        label_dict = {};
        with open(label_file_path, newline='') as csvfile :
            reader = csv.reader(csvfile, delimiter=',');

            for row in reader :

                label_dict[row[0]] = row[1];

        data_X_list = list();
        data_Y_list = list();
        for file_name in glob(os.path.join(data_file_path, '*.jpg')) :
            data_name = file_name.split('\\')[-1];

            data_name = data_name.split('.jpg')[0];
            try :
                label = label_dict[data_name];
            except :
                continue;
            jpgfile = Image.open(file_name);
            data_X_list.append(np.resize(np.asarray(jpgfile),(self.size,self.size,3)));
            label_np = np.zeros(shape=(2,));
            label_np[int(label)] = 1;
            data_Y_list.append(label_np);

        total_data_size = len(data_Y_list);

        perm = np.arange(total_data_size);
        np.random.shuffle(perm);

        data_X_np = np.stack(data_X_list);
        data_Y_np = np.stack(data_Y_list);

        shuffle_X = data_X_np[perm];
        shuffle_Y = data_Y_np[perm];

        test_ratio = 0.2;
        test_X = shuffle_X[:int(test_ratio*total_data_size)];
        test_Y = shuffle_Y[:int(test_ratio*total_data_size)];

        train_X = shuffle_X[int(test_ratio*total_data_size)+1:];
        train_Y = shuffle_Y[int(test_ratio*total_data_size)+1:];

        self.data_loader(train_X, train_Y, test_X, test_Y);

    def data_loader(self, X, Y, Test_filter_X=None, Test_filter_Y=None) :
        nd_x = np.stack(X);
        nd_y = np.stack(Y);

        if Test_filter_X is not None :
            nd_x_test = np.stack(Test_filter_X);
            nd_y_test = np.stack(Test_filter_Y);
            self.data = input_data.make_custom_data(nd_x,nd_y,nd_x_test, nd_y_test);
        else :
            self.data = input_data.make_custom_data(nd_x,nd_y);

    def reset_train_data(self, train_x, train_y):
        self.data = input_data.set_training_set(train_x,train_y, self.data.validation, self.data.test, reshape=False);

    def get_all_remain(self):
        remain_img, y = self.data.train.remain_all();
        return remain_img, y;

    def get_all(self):
        return self.data.train.get_all();

    def get_pool_data(self,num):
        while True :
            success = False;
            n_data_img, y = self.data.train.get_n_data(num);
            label = np.argmax(y, axis=1);
            for i in range(y.shape[1]) :
                if i in label :
                    success = True;
                    continue;
                else :
                    success = False;
                    break;
            if success :
                break;
        return n_data_img, y;


    def __call__(self, batch_size):
        batch_imgs, y = self.data.train.next_batch(batch_size)
        if self.flag == 'conv':
            return batch_imgs, y
        if self.is_tanh:
            batch_imgs = batch_imgs * 2 - 1
        return batch_imgs, y

    def test_batch_data(self, batch_size=-1):
        if batch_size != -1:
            batch_imgs = self.data.test._images[0:batch_size, :, :, :];
            batch_labels = self.data.test._labels[0:batch_size, :];
        else:
            batch_imgs = self.data.test._images;
            batch_labels = self.data.test._labels;

            batch_imgs = np.reshape(batch_imgs, (batch_imgs.shape[0], self.size_r, self.size_c, self.channel))

        return batch_imgs, batch_labels

    def train_batch_data(self, batch_size=-1):
        if batch_size != -1:
            batch_imgs = self.data.train._images[0:batch_size, :, :, :];
            batch_labels = self.data.train._labels[0:batch_size, :];
        else :
            batch_imgs = self.data.train._images;
            batch_labels = self.data.train._labels;

            batch_imgs = np.reshape(batch_imgs, (batch_imgs.shape[0], self.size_r, self.size_c, self.channel))

        return batch_imgs, batch_labels

    def test_custom(self, batch_size=-1):
        if batch_size != -1 :
            batch_imgs = self.data.test._images[0:batch_size, :, :, :];
            batch_labels = self.data.test._labels[0:batch_size, :];
        else :
            batch_imgs = self.data.test._images;
            batch_labels = self.data.test._labels;

        return batch_imgs, batch_labels;


class Oneshot_img_data():
    def __init__(self, flag='conv', is_tanh=False):
        #self.X_dim = sample_data.shape[0]*sample_data.shape[1]  # for mlp
        self.y_dim = 2  # class num    #### it is usually a regressor
        self.size = 256  # for conv
        #self.size_r = sample_data.shape[0];
        #self.size_c = sample_data.shape[1];
        self.channel = 1  # for conv



        self.flag = flag
        self.is_tanh = is_tanh

    def set_data(self, data_x, data_y):
        sample_img = data_x[0];

        self.size_r = sample_img.shape[0];
        self.size_c = sample_img.shape[1];

        self.data_loader(data_x, data_y, data_x, data_y);


    def data_loader(self, X, Y, Test_filter_X=None, Test_filter_Y=None) :
        nd_x = np.stack(X);
        nd_y = np.stack(Y);

        if Test_filter_X is not None :
            nd_x_test = np.stack(Test_filter_X);
            nd_y_test = np.stack(Test_filter_Y);
            self.data = input_data.make_custom_data(nd_x,nd_y,nd_x_test, nd_y_test);
        else :
            self.data = input_data.make_custom_data(nd_x,nd_y);

    def reset_train_data(self, train_x, train_y):
        self.data = input_data.set_training_set(train_x,train_y, self.data.validation, self.data.test, reshape=False);

    def get_all_remain(self):
        remain_img, y = self.data.train.remain_all();
        return remain_img, y;

    def get_all(self):
        return self.data.train.get_all();

    def get_pool_data(self,num):
        while True :
            success = False;
            n_data_img, y = self.data.train.get_n_data(num);
            label = np.argmax(y, axis=1);
            for i in range(y.shape[1]) :
                if i in label :
                    success = True;
                    continue;
                else :
                    success = False;
                    break;
            if success :
                break;
        return n_data_img, y;


    def __call__(self, batch_size):
        batch_imgs, y = self.data.train.next_batch(batch_size)
        if self.flag == 'conv':
            return batch_imgs, y
        if self.is_tanh:
            batch_imgs = batch_imgs * 2 - 1
        return batch_imgs, y

    def test_batch_data(self, batch_size=-1):
        if batch_size != -1:
            batch_imgs = self.data.test._images[0:batch_size, :, :, :];
            batch_labels = self.data.test._labels[0:batch_size, :];
        else:
            batch_imgs = self.data.test._images;
            batch_labels = self.data.test._labels;

            batch_imgs = np.reshape(batch_imgs, (batch_imgs.shape[0], self.size_r, self.size_c, self.channel))

        return batch_imgs, batch_labels

    def train_batch_data(self, batch_size=-1):
        if batch_size != -1:
            batch_imgs = self.data.train._images[0:batch_size, :, :, :];
            batch_labels = self.data.train._labels[0:batch_size, :];
        else :
            batch_imgs = self.data.train._images;
            batch_labels = self.data.train._labels;

            batch_imgs = np.reshape(batch_imgs, (batch_imgs.shape[0], self.size_r, self.size_c, self.channel))

        return batch_imgs, batch_labels

    def test_custom(self, batch_size=-1):
        if batch_size != -1 :
            batch_imgs = self.data.test._images[0:batch_size, :, :, :];
            batch_labels = self.data.test._labels[0:batch_size, :];
        else :
            batch_imgs = self.data.test._images;
            batch_labels = self.data.test._labels;

        return batch_imgs, batch_labels;


class vggface_data():
    def __init__(self, ) :
        self.dim = 68;
        pass

    def load_feature_data(self, train_num, class_idx,  data_root, seed) :

        sub_dirs = os.listdir(data_root);
        sub_dirs = sorted(sub_dirs)

        np.random.seed(seed);
        preprocess = transforms.Compose(
            [transforms.Resize((64, 64)),
             transforms.ToTensor(),
             # normalize
             ])

        target_class_file = sub_dirs[class_idx];
        # train data first
        image_files = sorted(os.listdir(data_root + target_class_file));


        total_x = list();
        total_y = list();
        for img in image_files :
            img_pil = preprocess(Image.open(data_root + target_class_file + "/" + img)).numpy();

            total_x.append(img_pil)
            total_y.append(class_idx)

        idx_shuffle = np.arange(len(total_x));
        np.random.shuffle(idx_shuffle);

        train_x = np.stack(total_x)[idx_shuffle[:train_num]];
        train_y = np.stack(total_y)[idx_shuffle[:train_num]];

        val_x = np.stack(total_x)[idx_shuffle[train_num:]];
        val_y = np.stack(total_y)[idx_shuffle[train_num:]];


        return train_x, train_y, val_x, val_y

class imagenet_data():
    def __init__(self, ) :
        self.dim = 68;
        pass

    def load_feature_data(self, train_num, class_idx,  data_root, seed) :

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        preprocess = transforms.Compose(
            [transforms.Resize((84, 84)),
             transforms.ToTensor(),
             #normalize
             ])

        sub_dirs = os.listdir(data_root);
        sub_dirs = sorted(sub_dirs)

        np.random.seed(seed);

        target_class_file = sub_dirs[class_idx];
        # train data first
        image_files = sorted(os.listdir(data_root + target_class_file));


        total_x = list();
        total_y = list();
        for img in image_files :
            img_pil = preprocess(Image.open(data_root + target_class_file +"/"+ img)).numpy();

            total_x.append(img_pil)
            total_y.append(class_idx)

        idx_shuffle = np.arange(len(total_x));
        np.random.shuffle(idx_shuffle);

        train_x = np.stack(total_x)[idx_shuffle[:train_num]];
        train_y = np.stack(total_y)[idx_shuffle[:train_num]];

        val_x = np.stack(total_x)[idx_shuffle[train_num:]];
        val_y = np.stack(total_y)[idx_shuffle[train_num:]];


        return train_x, train_y, val_x, val_y

class sentiment_data():
    def __init__(self, flag='mlp', is_tanh=False):
        self.x_dim = 4  # for mlp
        self.y_dim = 2  # class num    #### it is usually a regressor
        self.size = 64  # for conv

    def load_feature_data(self, train_num, val_num, test_num, max_len_pa):

        model_class = BertForSequenceClassification;
        model_class2 = BertModel;

        tokenizer_class = BertTokenizer;
        pretrained_weights = 'bert-base-uncased';

        tokenizer = tokenizer_class.from_pretrained(pretrained_weights);
        model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True, output_attentions=True, num_labels = 3);
        model2 = model_class2.from_pretrained(pretrained_weights);

        dict_label = dict();

        dict_label['very neg'] = 0
        dict_label['neg'] = 1
        dict_label['neu'] = 2
        dict_label['pos'] = 3
        dict_label['very pos'] = 4
        class_num = 5;

        model.eval();
        model2.eval();

        #model = KeyedVectors.load_word2vec_format('./data_sentiment/GoogleNews-vectors-negative300.bin', binary=True, limit=500000)
        max_line = -100;

        train_x_list, train_y_list, max_line = self.read_csv_data('./data_sentiment/sst5_train.csv', train_num, class_num, dict_label, model, model2, tokenizer, max_line)
        val_x_list, val_y_list, max_line = self.read_csv_data('./data_sentiment/sst5_val.csv', val_num, class_num, dict_label, model, model2, tokenizer, max_line)
        test_x_list, test_y_list, max_line = self.read_csv_data('./data_sentiment/sst5_test.csv', test_num, class_num, dict_label, model, model2, tokenizer, max_line)

        max_line = max_len_pa;  # hyperparameter

        #train_x_list = self.zero_pad_list(train_x_list, max_line);
        #val_x_list = self.zero_pad_list(val_x_list, max_line);
        #test_x_list = self.zero_pad_list(test_x_list, max_line);

        #train_x_list, pos_train_x_list = self.copy_pad_list(train_x_list, max_line);
        #val_x_list, pos_val_x_list = self.copy_pad_list(val_x_list, max_line);
        #test_x_list, pos_test_x_list = self.copy_pad_list(test_x_list, max_line);

        train_x_list = self.only_final_list(train_x_list);
        val_x_list = self.only_final_list(val_x_list);
        test_x_list = self.only_final_list(test_x_list);

        #return np.stack(train_x_list) + np.stack(pos_train_x_list), np.stack(train_y_list), np.stack(val_x_list)  + np.stack(pos_val_x_list), np.stack(val_y_list), np.stack(test_x_list)  + np.stack(pos_test_x_list), np.stack(test_y_list), max_line
        return np.stack(train_x_list), np.stack(train_y_list), np.stack(val_x_list), np.stack(val_y_list), np.stack(test_x_list), np.stack(test_y_list), max_line

    def read_csv_data(self, file_name, data_num, class_num, dict_label, model, model2, tokenizer, max_line):
        x_list = list();
        row_list = list();
        y_list = list();

        with open(file_name, newline='') as csvfile:
            train_reader = csv.reader(csvfile, delimiter=',');

            line_num = len(list(train_reader));
            csvfile.seek(0);
            line_idx = np.arange(line_num)
            np.random.shuffle(line_idx);
            line_idx = line_idx[:data_num];
            file_idx = -1;
            line_info = list()

            for row in train_reader:
                file_idx+=1;
                if file_idx not in line_idx :
                    continue;

                label = np.zeros(shape=(class_num,));
                label[dict_label[row[1]]] = 1;

                if max_line < len(row[0]):
                    max_line = len(row[0]);

                line_info.append(len(row[0]));
                row_list.append(torch.tensor([tokenizer.encode(row[0].lower())]));
                y_list.append(label);

            for row_part in row_list :
                with torch.no_grad():
                    last_hidden_states, last_attention = model(row_part)[-2:];
                    last_hidden_states = model2(row_part)[0];
                #wordvec_tensor = list(last_hidden_states);
                wordvec_tensor = last_hidden_states;
                x_list.append(wordvec_tensor.squeeze().numpy());


        return x_list, y_list, max_line;

    def last_pad_list(self, list_x, max_line) :
        pad_result = list();

        for x in list_x :
            if x.shape[0] < max_line :
                #zero_matrix = np.zeros(shape=(max_line - x.shape[0], x.shape[1]));
                zero_repeat = np.tile(x[-1, :], (int(max_line - x.shape[0]), 1));
                #zero_matrix = np.ones(shape=(max_line - x.shape[0], x.shape[1]));
                pad_result.append(np.concatenate([x,zero_repeat],axis=0));
            else :
                x = x[:max_line]
                pad_result.append(x);
        return pad_result;

    def only_final_list(self, list_x) :
        pad_result = list();

        for x in list_x :
            pad_result.append(x[-1,:]);

        return pad_result;

    def zero_pad_list(self, list_x, max_line) :
        pad_result = list();

        for x in list_x :
            if x.shape[0] < max_line :
                zero_matrix = np.zeros(shape=(max_line - x.shape[0], x.shape[1]));
                #zero_repeat = np.tile(x[-1, :], (int(max_line - x.shape[0]), 1));
                zero_matrix = np.ones(shape=(max_line - x.shape[0], x.shape[1]));
                pad_result.append(np.concatenate([x,zero_matrix],axis=0));
            else :
                x = x[:max_line]
                pad_result.append(x);
        return pad_result;

    def copy_pad_list(self, list_x, max_line) :
        pad_result = list();
        position_result = list();
        for x in list_x :
            x_repeat = np.tile(x, (int(max_line // x.shape[0]) + 1,1))
            position_vec = np.repeat(np.expand_dims(np.arange(x.shape[0]) / max_line, 1), x.shape[1], axis=1);
            position_repeat = np.tile(position_vec, (int(max_line // x.shape[0]) +1,1));
            if x.shape[0] < max_line :
                pad_result.append(x_repeat[:max_line]);
                position_result.append(position_repeat[:max_line]);
            else :
                pad_result.append(x[:max_line]);
                position_result.append(position_vec[:max_line]);

        return pad_result, position_result;


class bcws_data():
    def __init__(self, flag='mlp', is_tanh=False):
        self.x_dim = 4  # for mlp
        self.y_dim = 2  # class num    #### it is usually a regressor
        self.size = 64  # for conv

    def load_feature_data(self, train_num, val_num, test_num):

        return self.read_csv_data('./data_BC_wis/breast-cancer-wisconsin.data', train_num, val_num, test_num)

    def read_csv_data(self, file_name, train_num, val_num, test_num):


        total_list_x_1 = list();
        total_list_y_1 = list();
        total_list_x_2 = list();
        total_list_y_2 = list();


        with open(file_name) as csvfile:
            lines = csvfile.readlines();

            np.random.shuffle(lines);

            line_idx = 0;
            for line in lines:
                line = line.split('\n')[0];
                values = line.split(',');

                x_val = np.zeros(shape=[9,1]);
                target = np.zeros(shape=[2,]);
                fail = False;
                for i in range(1,10) :

                    try :
                        x_val[i-1] = float(values[i]);
                    except :
                        fail= True;
                        break;
                if fail :
                    continue;

                if values[10] == '2' :
                    target[0] = 1;
                    total_list_x_1.append(x_val);
                    total_list_y_1.append(target);

                elif values[10] == '4' :
                    target[1] = 1;
                    total_list_x_2.append(x_val);
                    total_list_y_2.append(target);


                line_idx +=1;

        bal_len = len(total_list_x_2);
        cut_point = int(bal_len* train_num);

        train_x_list = total_list_x_1[:cut_point] + total_list_x_2[:cut_point];
        train_y_list = total_list_y_1[:cut_point] + total_list_y_2[:cut_point];

        val_x_list = total_list_x_1[cut_point:] + total_list_x_2[cut_point:];
        val_y_list = total_list_y_1[cut_point:] + total_list_y_2[cut_point:];
        test_x_list = val_x_list;
        test_y_list = val_y_list;
        np_train_x = np.stack(train_x_list);
        np_val_x = np.stack(val_x_list);
        np_test_x = np.stack(test_x_list);

        mean_x = np.mean(np_train_x, 0);
        std_x = np.std(np_train_x,0);

        np_train_x = (np_train_x - mean_x)/std_x;
        np_val_x = (np_val_x - mean_x)/std_x;
        np_test_x = (np_test_x - mean_x)/std_x;


        return np_train_x, np.stack(train_y_list), np_val_x, np.stack(val_y_list), np_test_x, np.stack(test_y_list);


class bcws_diag_data():
    def __init__(self, flag='mlp', is_tanh=False):
        self.x_dim = 4  # for mlp
        self.y_dim = 2  # class num    #### it is usually a regressor
        self.size = 64  # for conv

    def load_feature_data(self, train_num, val_num, test_num):

        return self.read_csv_data('./data_BC_wis_diagnostic/BC_wis_dat.csv', train_num, val_num, test_num)

    def read_csv_data(self, file_name, train_num, val_num, test_num):

        train_x_list = list();
        val_x_list = list();
        test_x_list = list();

        train_y_list = list();
        val_y_list = list();
        test_y_list = list();

        with open(file_name) as csvfile:
            lines = csvfile.readlines();

            np.random.shuffle(lines);

            line_idx = 0;
            for line in lines:
                line = line.split('\n')[0];
                values = line.split(',');

                x_val = np.zeros(shape=[30,1]);
                target = np.zeros(shape=[2,]);
                for i in range(1,31) :

                    x_val[i-1] = float(values[i]);
                target[int(values[0])] = 1;

                if line_idx < train_num :
                    train_x_list.append(x_val);
                    train_y_list.append(target);
                else :
                    val_x_list.append(x_val);
                    val_y_list.append(target);
                line_idx +=1;

        test_x_list = val_x_list;
        test_y_list = val_y_list;
        np_train_x = np.stack(train_x_list);
        np_val_x = np.stack(val_x_list);
        np_test_x = np.stack(test_x_list);

        mean_x = np.mean(np_train_x, 0);
        std_x = np.std(np_train_x,0);

        np_train_x = (np_train_x - mean_x)/std_x;
        np_val_x = (np_val_x - mean_x)/std_x;
        np_test_x = (np_test_x - mean_x)/std_x;


        return np_train_x, np.stack(train_y_list), np_val_x, np.stack(val_y_list), np_test_x, np.stack(test_y_list);

class weather_data():
    def __init__(self, flag='mlp', is_tanh=False):
        self.x_dim = 4  # for mlp
        self.y_dim = 2  # class num    #### it is usually a regressor
        self.size = 64  # for conv

    def load_feature_data(self, train_num, val_num, test_num, region = 'Manaus'):

        return self.read_csv_data('./data_weather/' + region + '.txt', train_num, val_num, test_num)

    def read_csv_data(self, file_name, train_num, val_num, test_num):

        train_x_list = list();
        val_x_list = list();
        test_x_list = list();

        train_y_list = list();
        val_y_list = list();
        test_y_list = list();

        with open(file_name) as tsvfile:
            lines = tsvfile.readlines();

            np.random.shuffle(lines);

            line_idx = 0;
            for line in lines:
                line = line.split('\n')[0];
                values = line.split('\t');

                x_val = np.zeros(shape=[5,1]);
                target = np.zeros(shape=[1,]);
                for i in range(5) :
                    x_val[i] = float(values[i]);
                target[0] = values[5];

                if line_idx < train_num :
                    train_x_list.append(x_val);
                    train_y_list.append(target);
                elif line_idx < train_num+val_num:
                    val_x_list.append(x_val);
                    val_y_list.append(target);
                elif line_idx < train_num+ val_num+ test_num :
                    test_x_list.append(x_val);
                    test_y_list.append(target);
                line_idx +=1;


        return np.stack(train_x_list), np.stack(train_y_list), np.stack(val_x_list), np.stack(val_y_list), np.stack(test_x_list), np.stack(test_y_list);



class example_toy_img2() :
    def __init__(self, flag='mlp', is_tanh=False):
        self.x_dim = 4  # for mlp
        self.y_dim = 2  # class num    #### it is usually a regressor
        self.size = 64  # for conv

    def draw_img(self, idx, x,y, x_vec, x_vec_img):

        if idx == 0:

            circle_radius = np.random.uniform(low=10.0, high=30.0, size=(1, 1));
            rr, cc = draw.circle(int(x[0]), int(y[0]), radius=int(circle_radius), shape=x_vec.shape)

            x_vec_img[rr, cc, :] = 255;

        elif idx == 1:

            rect_radius_x, rect_radius_y = np.random.uniform(low=10.0, high=30.0, size=(2, 1));
            rr, cc = draw.rectangle((int(x[0]), int(y[0])),
                                    end=(int(x[0]) + int(rect_radius_x), int(y[0]) + int(rect_radius_y)),
                                    shape=x_vec.shape)

            x_vec_img[rr, cc, :] = 255;

        elif idx == 2:
            x[np.argmin(x)] -=10;
            x[np.argmax(x)] +=10;

            y[np.argmin(y)] -=10;
            y[np.argmax(y)] +=10;
            rr, cc = draw.polygon(x, y);
            x_vec_img[rr, cc, :] = 255;

        elif idx == 3:

            elipse_radius_x, elipse_radius_y = np.random.uniform(low=10.0, high=30.0, size=(2, 1));
            rotation_val = np.random.uniform(low=0.0, high=90.0, size=(1, 1));
            rr, cc = draw.ellipse(int(x[0]), int(y[0]), int(elipse_radius_x), int(elipse_radius_y), shape=x_vec.shape,
                                  rotation=float(rotation_val));
            x_vec_img[rr, cc, :] = 255;


        return x_vec_img;

    def save_example_task_img(self, pool_num) :
        example_vector_x_list = list();
        example_vector_y_list = list();

        for i in range(pool_num):
            y_vec = np.zeros(shape=[2, ]);
            x_vec = np.zeros(shape=[256, 256]);
            x_vec_img = np.zeros(shape=[256, 256, 3]);

            # 0 for circle, 1 for rectangle, 2 for polygon
            all_possb = np.tile(np.arange(3), 4);

            if i%2 == 0 :
                while True :
                    np.random.shuffle(all_possb);
                    assignment = all_possb[:4];

                    if (((assignment[1] == 0 and assignment[0] == 2) or (assignment[3] == 0 and assignment[2] == 2))) and np.where(assignment==2)[0].shape[0] == 2 :

                        y_vec[0] = 1;
                        break;
                    else :
                        continue;
            else :
                while True :
                    np.random.shuffle(all_possb);
                    assignment = all_possb[:4];

                    if (((assignment[1] == 0 and assignment[0] == 2) or (assignment[3] == 0 and assignment[2] == 2))) and np.where(assignment==2)[0].shape[0] == 2 :
                        continue;
                    else :
                        y_vec[1] = 1;
                        break;

            for place_idx in range(4) :
                if place_idx == 0 :
                    x, y = np.random.uniform(low=25.0, high=100.0, size=(2, 3));
                elif place_idx ==1 :
                    x = np.random.uniform(low=25.0, high=100.0, size=(3, 1));
                    y = np.random.uniform(low=153.0, high=231.0, size=(3, 1));
                elif place_idx == 2:
                    x = np.random.uniform(low=153.0, high=231.0, size=(3, 1));
                    y = np.random.uniform(low=25.0, high=100.0, size=(3, 1));
                elif place_idx ==3:
                    x, y = np.random.uniform(low=153.0, high=231.0, size=(2, 3));


                x_vec_img = self.draw_img(assignment[place_idx], x, y, x_vec, x_vec_img);

            #plt.imsave('here.png',x_vec_img);
            example_vector_x_list.append(Image.fromarray(x_vec_img.astype('uint8'), 'RGB'));
            example_vector_y_list.append(y_vec)

        return example_vector_x_list, np.stack(example_vector_y_list);



class example_toy_img() :
    def __init__(self, flag='mlp', is_tanh=False):
        self.x_dim = 4  # for mlp
        self.y_dim = 2  # class num    #### it is usually a regressor
        self.size = 64  # for conv

    def save_example_task_img(self, pool_num) :
        example_vector_x_list = list();
        example_vector_y_list = list();

        for i in range(pool_num):
            y_vec = np.zeros(shape=[2, ]);
            x_vec = np.zeros(shape=[256, 256]);
            x_vec_img = np.zeros(shape=[256, 256, 3]);



            if i % 2 == 0:

                polygon_x, polygon_y = np.random.uniform(low=25.0, high=100.0, size=(2, 3));
                rect_x, rect_y = np.random.uniform(low=153.0, high=231.0, size=(2, 1));

                rect_radius_x, rect_radius_y = np.random.uniform(low=10.0, high=30.0, size=(2, 1));

                poly_rr, poly_cc = draw.polygon( polygon_x, polygon_y)
                rect_rr, rect_cc = draw.rectangle((int(rect_x),int(rect_y)), end=(int(rect_x) + int(rect_radius_x), int(rect_y) + int(rect_radius_y)), shape=x_vec.shape)

                x_vec_img[poly_rr, poly_cc, :] = 255;
                x_vec_img[rect_rr, rect_cc, :] = 255;


                y_vec[0] = 1;

            else:

                circle_x, circle_y = np.random.uniform(low=25.0, high=100.0, size=(2, 1));
                elipse_x, elipse_y = np.random.uniform(low=153.0, high=231.0, size=(2, 1));

                circle_radius = np.random.uniform(low=10.0, high=30.0, size=(1,1));
                elipse_radius_x, elipse_radius_y = np.random.uniform(low=10.0, high=30.0, size=(2, 1));

                rotation_val = np.random.uniform(low=0.0, high=90.0, size=(1,1));

                circle_rr, circle_cc = draw.circle(int(circle_x),int(circle_y), radius = int(circle_radius), shape=x_vec.shape)
                elipse_rr, elipse_cc = draw.ellipse(int(elipse_x),int(elipse_y), int(elipse_radius_x), int(elipse_radius_y), shape=x_vec.shape, rotation = float(rotation_val));

                x_vec_img[circle_rr, circle_cc,:] = 255
                x_vec_img[elipse_rr, elipse_cc,:] = 255


                y_vec[1] = 1;

            example_vector_x_list.append(Image.fromarray(x_vec_img.astype('uint8'), 'RGB'));
            example_vector_y_list.append(y_vec)

        return example_vector_x_list, np.stack(example_vector_y_list);


class example_toy1_data() :

    def __init__(self, flag='mlp', is_tanh=False):
        self.x_dim = 4  # for mlp
        self.y_dim = 2  # class num    #### it is usually a regressor
        self.size = 64  # for conv

    def save_example_task_vector(self, pool_num):
        example_vector_x_list = list();
        example_vector_y_list = list();

        for i in range(pool_num) :
            y_vec = np.zeros(shape=[2,]);
            x_vec = np.zeros(shape=[4,]);
            if i%2 ==0 :
                while True :
                    x, y = np.random.uniform(low=0.0, high=1.5, size=(2,1));
                    c, b = np.random.uniform(low=0.0, high=1.5, size=(2,1));

                    x_y = np.sin(x+y);
                    c_b = np.cos(c+b);

                    if (x_y-0.2) > (c_b) :
                        x_vec = np.stack([x,y,c,b]);
                        y_vec[0] = 1;
                        break;
                    else :
                        continue;
            else :
                while True :
                    x, y = np.random.uniform(low=0.0, high=1.5, size=(2,1));
                    c, b = np.random.uniform(low=0.0, high=1.5, size=(2,1));

                    x_y = np.sin(x + y);
                    c_b = np.cos(c + b);

                    if (x_y+0.2)< (c_b) :
                        x_vec = np.stack([x,y,c,b]);
                        y_vec[1] = 1;
                        break;

                    else :
                        continue;
            example_vector_x_list.append(x_vec);
            example_vector_y_list.append(y_vec)

        return np.stack(example_vector_x_list), np.stack(example_vector_y_list);



class example1_data():
    def __init__(self, flag='mlp', is_tanh=False):
        self.x_dim = 4  # for mlp
        self.y_dim = 2  # class num    #### it is usually a regressor
        self.size = 64  # for conv

    def save_example_task_vector(self, pool_num):
        example_vector_x_list = list();
        example_vector_y_list = list();

        for i in range(pool_num) :
            y_vec = np.zeros(shape=[2,]);
            x_vec = np.zeros(shape=[4,]);
            if i%2 ==0 :
                while True :
                    x, y = np.random.uniform(low=0.0, high=1.5, size=(2,1));
                    c, b = np.random.uniform(low=0.0, high=1.5, size=(2,1));

                    if (x+y-0.2) > (c+b) :
                        x_vec = np.stack([x,y,c,b]);
                        y_vec[0] = 1;
                        break;
                    else :
                        continue;
            else :
                while True :
                    x, y = np.random.uniform(low=0.0, high=1.5, size=(2,1));
                    c, b = np.random.uniform(low=0.0, high=1.5, size=(2,1));

                    if (x+y+0.2)< (c+b) :
                        x_vec = np.stack([x,y,c,b]);
                        y_vec[1] = 1;
                        break;

                    else :
                        continue;
            example_vector_x_list.append(x_vec);
            example_vector_y_list.append(y_vec)

        return np.stack(example_vector_x_list), np.stack(example_vector_y_list);


class example2_data():
    def __init__(self, flag='mlp', is_tanh=False):
        self.x_dim = 4  # for mlp
        self.y_dim = 2  # class num    #### it is usually a regressor
        self.size = 64  # for conv

    def save_example_task_vector(self, pool_num):
        example_vector_x_list = list();
        example_vector_y_list = list();

        for i in range(pool_num) :
            y_vec = np.zeros(shape=[2,]);
            x_vec = np.zeros(shape=[4,]);
            if i%2 ==0 :
                while True :
                    x, y, c, b = np.random.uniform(low=0.0, high=1.0, size=(4,1));

                    if x+y+c+b > 1.5 :
                        x_vec = np.stack([x,y,c,b]);
                        y_vec[0] = 1;
                        break;
                    else :
                        continue;
            else :
                while True :
                    x, y, c, b = np.random.uniform(low=0.0, high=1.0, size=(4,1));

                    if x+y+c+b < 1.5 :
                        x_vec = np.stack([x,y,c,b]);
                        y_vec[1] = 1;
                        break;

                    else :
                        continue;
            example_vector_x_list.append(x_vec);
            example_vector_y_list.append(y_vec)

        return np.stack(example_vector_x_list), np.stack(example_vector_y_list);


class example3_data():
    def __init__(self, flag='mlp', is_tanh=False):
        self.x_dim = 4  # for mlp
        self.y_dim = 2  # class num    #### it is usually a regressor
        self.size = 64  # for conv

    def save_example_task_vector(self, pool_num):
        example_vector_x_list = list();
        example_vector_y_list = list();

        for i in range(pool_num) :
            y_vec = np.zeros(shape=[2,]);
            x_vec = np.zeros(shape=[4,]);
            if i%2 ==0 :
                while True :
                    x, y, c, b = np.random.uniform(low=0.0, high=1.0, size=(4,1));

                    if x+y+c+b > 1.5 :
                        x_vec = np.stack([x,y,c,b]);
                        y_vec[0] = 1;
                        break;
                    else :
                        continue;
            else :
                while True :
                    x, y, c, b = np.random.uniform(low=0.0, high=1.0, size=(4,1));

                    if x+y+c+b < 1.5 :
                        x_vec = np.stack([x,y,c,b]);
                        y_vec[1] = 1;
                        break;

                    else :
                        continue;
            example_vector_x_list.append(x_vec);
            example_vector_y_list.append(y_vec)

        return np.stack(example_vector_x_list), np.stack(example_vector_y_list);


class OneShot_data():
    def __init__(self, flag='mlp', is_tanh=False):
        self.x_dim = 21  # for mlp
        self.y_dim = 2  # class num    #### it is usually a regressor
        self.size = 64  # for conv
        #self.size_r = sample_data.shape[0];
        #self.size_c = sample_data.shape[1];
        self.channel = 1  # for conv

        self.batch_size = -1;
        self.flag = flag
        self.is_tanh = is_tanh
        self.discrete_val_1_2 = [0.5, 1, 1.5];
        self.discrete_val_other = [1, 2, 3, 4, 5];

        self.img_row = 25;
        self.img_col = 50;

    def save_gen_task_vector(self, pool_num):

        task_vector_list = list();
        target_list = list();
        for i in range(pool_num) :
            y = np.zeros(shape=[2,1]);
            if i%2 == 0 :
                task_vector, task_vector_raw = self.rand_gen_task_vector(mode=0);
                y[0] = 1;
            else :
                task_vector, task_vector_raw = self.rand_gen_task_vector(mode=1);
                y[1] = 1;


            task_vector_list.append(task_vector);
            target_list.append(np.squeeze(y));


        task_vector_arry = np.stack(task_vector_list);


        return task_vector_arry, target_list;


    def read_from_file(self, file_name_img, file_name_vector):
        vector_mat = scipy.io.loadmat(file_name_vector);
        vector_arry = vector_mat['vector'];

        img_mat = scipy.io.loadmat(file_name_img);
        img_arry = img_mat['img'];

        return vector_arry, img_arry;


    def rand_gen_task_vector(self, mode = 0):

        '''
            randomly populate task vector.
            before this, you need to design task vector

            1 => time interval between each stimulus (relative to stimulus duration)        [0.25 0.5 1 1.5 2]
            2 => time interval between stimulus and reward (relative to stimulus duration)  [0.25 0.5 1 1.5 2]
            3,4 => location of image 3 (x,y)                   x : [1 2 3 4 5], y : [1 2 3 4 5]
            5,6 => location of image 2_1 (x,y)
            7,8 => location of image 2_2 (x,y)
            9,10 => location of image 2_3 (x,y)
            11,12 => location of image 2_4 (x,y)
            13,14 => location of image 2_5 (x,y)
            15,16 => location of image 2_6 (x,y)
            17,18 => location of image 2_7 (x,y)
            19,20 => location of image 2_8 (x,y)
            21 => location of reward novel (y)

        '''

        task_vector = np.zeros(shape=(21,));
        task_vector_raw = np.zeros(shape=(21,));


        rand_val1 = randint(0,len(self.discrete_val_1_2)-1);
        #rand_val1 = 0;

        task_vector[0] = (self.discrete_val_1_2[rand_val1] - min(self.discrete_val_1_2))/(max(self.discrete_val_1_2) - min(self.discrete_val_1_2));
        task_vector_raw[0] = self.discrete_val_1_2[rand_val1];

        rand_val2 = randint(0,len(self.discrete_val_1_2)-1);
        #rand_val2 = 0;
        task_vector[1] = (self.discrete_val_1_2[rand_val2] - min(self.discrete_val_1_2))/(max(self.discrete_val_1_2) - min(self.discrete_val_1_2));
        task_vector_raw[1] = self.discrete_val_1_2[rand_val2];

        row = np.repeat(np.arange(5), 5, axis=0);
        col = np.tile(np.arange(5), 5);
        perm0 = np.arange(row.shape[0]);
        np.random.shuffle(perm0);
        perm_idx = 0;
        for i in range(2,task_vector.shape[0]-1,2) :
            task_vector[i] = (self.discrete_val_other[row[perm0[perm_idx]]] - min(self.discrete_val_other)) / (max(self.discrete_val_other) - min(self.discrete_val_other));
            task_vector[i+1] = (self.discrete_val_other[col[perm0[perm_idx]]] - min(self.discrete_val_other)) / (max(self.discrete_val_other) - min(self.discrete_val_other));
            task_vector_raw[i] = self.discrete_val_other[row[perm0[perm_idx]]];
            task_vector_raw[i+1] = self.discrete_val_other[col[perm0[perm_idx]]];
            perm_idx +=1;

        if mode == 0 :
            while True :
                rand_val3 = randint(0,len(self.discrete_val_other)-1);
                if rand_val3 == row[perm0[0]] :
                    continue;
                else :
                    break;

            task_vector[20] = (self.discrete_val_other[rand_val3] - min(self.discrete_val_other))/(max(self.discrete_val_other) - min(self.discrete_val_other));
            task_vector_raw[20] = self.discrete_val_other[rand_val3];

        else :
            rand_val3 = row[perm0[0]];
            task_vector[20] = (self.discrete_val_other[rand_val3] - min(self.discrete_val_other))/(max(self.discrete_val_other) - min(self.discrete_val_other));
            task_vector_raw[20] = self.discrete_val_other[rand_val3];

        return task_vector, task_vector_raw;

class OneShot_Word_data(OneShot_data):
    def __init__(self, flag='mlp', is_tanh=False):
        self.x_dim = 21  # for mlp
        self.y_dim = 2  # class num    #### it is usually a regressor
        self.size = 64  # for conv
        #self.size_r = sample_data.shape[0];
        #self.size_c = sample_data.shape[1];
        self.channel = 1  # for conv

        self.batch_size = -1;
        self.flag = flag
        self.is_tanh = is_tanh
        self.discrete_val_1_2 = [0.5, 1, 1.5];
        self.discrete_val_other = [1, 2, 3, 4, 5];

        self.img_row = 25;
        self.img_col = 50;

    def save_gen_task_vector(self, pool_num):

        task_vector_list = list();
        target_list = list();
        task_vector_word_list = list();

        for i in range(pool_num) :
            y = np.zeros(shape=[2,1]);
            if i%2 == 0 :
                task_vector, task_vector_raw = self.rand_gen_task_vector(mode=0);
                y[0] = 1;
            else :
                task_vector, task_vector_raw = self.rand_gen_task_vector(mode=1);
                y[1] = 1;


            task_word = self.task_vector_to_word_vector(task_vector_raw);

            task_word=np.squeeze(task_word);
            task_vector_list.append(task_vector);
            target_list.append(np.squeeze(y));
            task_vector_word_list.append(task_word);


        task_vector_word_arry = np.stack(task_vector_word_list);


        return task_vector_word_arry, target_list;


    def task_vector_to_word_vector(self, task_vector):

        size_of_stim = 2 + 1 + 1*8 + 1 + 1+ 1; # 7 -> 2(time_interval-stim , time interval line)+ 1(stim 3) +8 (stim2) + 1 (stim1) + 1 (gen_reward) + 1 (novel_reward) = 14

        idx_stim_interval = 0;
        idx_line_interval = 1;
        idx_3 = 2;
        idx_2 = 3;
        idx_1 = 11;
        idx_gen_reward = 12;
        idx_nov_reward = 13


        total_stim_list = list();

        for i in range(5) :  # for 5 stimulus
            for j in range(5) : # for 5 column
                stim_value = np.zeros(shape=(size_of_stim,1), dtype= float);

                is_2or3 = False;
                for rand_vector_idx in range(2,20,2) :
                    if i == task_vector[rand_vector_idx]-1 and j == task_vector[rand_vector_idx+1]-1 :
                        if rand_vector_idx == 2 : # 3
                            stim_value[idx_3] = 1.0;
                            total_stim_list.append(stim_value);
                            is_2or3 = True;
                            break;
                        else: #2
                            stim_value[idx_2] = 1.0;
                            idx_2 += 1;
                            total_stim_list.append(stim_value);
                            is_2or3 = True;
                            break;
                if is_2or3 == False :
                    stim_value[idx_1] = 1.0;
                    total_stim_list.append(stim_value);

                stim_interval = np.zeros(shape=(size_of_stim,1), dtype= float);
                stim_interval[idx_stim_interval] = 1.0;
                total_stim_list.append(stim_interval);

            reward_value = np.zeros(shape=(size_of_stim,1), dtype = float);

            if i == task_vector[20]-1 :
                reward_value[idx_nov_reward] = 1.0;
                total_stim_list.append(reward_value);
            else :
                reward_value[idx_gen_reward] = 1.0;
                total_stim_list.append(reward_value);

            if i != 4 :
                line_interval = np.zeros(shape=(size_of_stim,1), dtype= float);
                line_interval[idx_line_interval] = 1.0;
                total_stim_list.append(line_interval);

        return np.stack(total_stim_list);



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

if __name__ == '__main__':
    data = face3D()
    print (data(17).shape)
