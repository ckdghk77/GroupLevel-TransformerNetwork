
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse

import os
matplotlib.use('Agg')

parser = argparse.ArgumentParser()

parser.add_argument('--load-folder', type=str, default='./out')
parser.add_argument('--is-rgb', type=bool, default=False);

args = parser.parse_args()



file_generated = os.path.join(args.load_folder,"generated.npy");
file_train = os.path.join(args.load_folder,"train_x.npy");

generated = np.load(file_generated);
train_x = np.load(file_train);

plt.rcParams["figure.figsize"] = (7.85, 7.9);
fig_count = 0;
for src in range(generated.shape[0]) :

    for src in range(generated.shape[0]):  # src

        for tar in range(generated.shape[2]):  # target
            if tar == 0:
                continue;

            total_row_num = generated.shape[0];
            total_col_num = generated.shape[0];

            fig, axes = plt.subplots(total_row_num,
                                     total_col_num,
                                     gridspec_kw={'wspace': 0, 'hspace': 0});

            for idx in range(generated.shape[1] + 2):

                cur_row = idx // total_row_num;
                cur_col = idx % total_row_num;
                axes[cur_row, cur_col].axis('off')
                if idx == 0:  # src_img
                    if args.is_rgb :
                        axes[cur_row, cur_col].imshow(np.transpose(train_x[src, 0], [1, 2, 0]));
                    else :
                        axes[cur_row, cur_col].imshow(np.squeeze(train_x[src, 0]),cmap='Greys');


                elif idx == generated.shape[1] + 1:
                    if args.is_rgb:
                        axes[cur_row, cur_col].imshow(np.transpose(train_x[src, tar], [1, 2, 0]));
                    else :
                        axes[cur_row, cur_col].imshow(np.squeeze(train_x[src, tar]),cmap='Greys');
                else:
                    if args.is_rgb:
                        axes[cur_row, cur_col].imshow(
                            np.transpose(generated[src, idx - 1, tar], [1, 2, 0]));
                    else :
                        axes[cur_row, cur_col].imshow(np.squeeze(generated[src, idx - 1, tar]), cmap='Greys');

            plt.axis('off')
            #plt.savefig("result.jpg", format='jpg', dpi=1200)
            plt.savefig(os.path.join(args.load_folder,"result" +str(fig_count)+ ".jpg"), format='jpg')
            plt.close('all')
            fig_count+=1;
            #plt.show()








