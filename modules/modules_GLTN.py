import torch.nn as nn

from utils import *

from torch.autograd import Variable
from modules.layers import SLP, MLP
from torch_deform_conv.deform_conv import th_batch_map_offsets , th_generate_grid


_EPS = 1e-10


class Transformer(nn.Module):

    def __init__(self, input_size, hidden_size, transform_hidden, sequence_len,  type='mnist_conv'):
        super(Transformer, self).__init__()
        self.input_size = input_size;
        self.hidden_size = hidden_size;
        self.transform_hidden = transform_hidden;
        self.sequence_len = sequence_len;
        self.sparse = False;
        self.type = type;
        self._grid_param = None

        if type == 'mnist_conv':

            self.mlp1 = nn.Sequential(
                nn.Conv2d(1, 16, 4, 2, 0),
                nn.LeakyReLU(),
                nn.Conv2d(16, 8, 4, 2, 0),
                nn.LeakyReLU(),
                nn.Conv2d(8, 4, 3, 1, 0),
                nn.LeakyReLU(),
            )

        elif type == 'omniglot_conv':
            self.mlp1 = nn.Sequential(
                nn.Conv2d(1, 32, 4, 2, 0),
                nn.LeakyReLU(),
                nn.Conv2d(32, 16, 4, 2, 0),
                nn.LeakyReLU(),
                nn.Conv2d(16, 8, 3, 1, 0),
                nn.LeakyReLU(),
            )
        elif type == 'fmnist_conv':
            self.mlp1 = nn.Sequential(
                nn.Conv2d(1, 32, 4, 2, 0),
                nn.LeakyReLU(),
                nn.Conv2d(32, 16, 4, 2, 0),
                nn.LeakyReLU(),
                nn.Conv2d(16, 8, 3, 1, 0),
                nn.LeakyReLU(),
            )

        elif type == 'cifar_conv':
            self.mlp1 = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 0),
                nn.LeakyReLU(),
                nn.Conv2d(64, 32, 4, 1, 0),
                nn.LeakyReLU(),
                nn.Conv2d(32, 16, 4, 1, 0),
                nn.LeakyReLU(),
                nn.Conv2d(16, 16, 4, 1, 0),
                nn.LeakyReLU(),
                nn.Conv2d(16, 16, 3, 1, 0),
                nn.LeakyReLU(),
            )
        elif type == 'custom_gray_conv':
            self.mlp1 = nn.Sequential(
                nn.Conv2d(1, 16, 4, 2, 0),
                nn.LeakyReLU(),
                nn.Conv2d(16, 32, 4, 2, 0),
                nn.LeakyReLU(),
                nn.Conv2d(32, 32, 4, 2, 0),
                nn.LeakyReLU(),
                nn.Conv2d(32, 32, 3, 1, 0),
                nn.LeakyReLU(),
            )

        elif type == "custom_rgb_conv" or type=="custom_heavy_rgb_conv":
            self.mlp1 = nn.Sequential(
                nn.Conv2d(3, 16, 4, 2, 0),
                nn.LeakyReLU(),
                nn.Conv2d(16, 32, 4, 2, 0),
                nn.LeakyReLU(),
                nn.Conv2d(32, 32, 4, 2, 0),
                nn.LeakyReLU(),
                nn.Conv2d(32, 32, 3, 1, 0),
                nn.LeakyReLU(),
            )


        if type == 'mnist_conv':

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(self.hidden_size, 32, 7, 1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(32, 16, 3, 1, padding=1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(16, 8, 4, 2, padding=1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(8, 1, 4, 2, padding=1),
                nn.Tanh()

            )

            self.transformer = nn.Sequential(
                nn.ConvTranspose2d(self.hidden_size + self.transform_hidden, 16, 4, 1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(16, 8, 4, 1, padding=1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(8, 2, 3, 1, padding=1),
            )

            self.channel_transformer_conv = nn.ModuleList([nn.Sequential(
                nn.Conv2d(self.transform_hidden + 1, 8, 1, 1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.Conv2d(8, 4, 1, 1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.Conv2d(4, 1, 1, 1, bias=False),
                nn.Tanh()

            ) for _ in range(3)])  # 3 here is a channel size


        elif type == 'omniglot_conv':

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(self.hidden_size, 32, 7, 1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(32, 16, 3, 2, padding=1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(16, 8, 4, 2, padding=1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(8, 8, 4, 1, padding=1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(8, 1, 4, 1, padding=1),
                nn.Tanh()
            )

            self.transformer = nn.Sequential(
                nn.ConvTranspose2d(self.hidden_size + self.transform_hidden, 32, 4, 1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(32, 16, 4, 1, padding=1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(16, 8, 4, 1, padding=1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(8, 2, 3, 1, padding=1),
            )

            self.channel_transformer_conv = nn.ModuleList([nn.Sequential(
                nn.Conv2d(self.transform_hidden + 1, 8, 1, 1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.Conv2d(8, 4, 1, 1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.Conv2d(4, 1, 1, 1, bias=False),
                nn.Tanh()

            ) for _ in range(3)])  # 3 here is a channel size

        elif type == 'fmnist_conv':

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(self.hidden_size, 32, 7, 1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(32, 16, 3, 2, padding=1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(16, 8, 4, 2, padding=1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(8, 8, 4, 1, padding=1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(8, 1, 4, 1, padding=1),
                nn.Tanh()
            )

            self.transformer = nn.Sequential(
                nn.ConvTranspose2d(self.hidden_size + self.transform_hidden, 32, 4, 1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(32, 16, 4, 1, padding=1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(16, 8, 4, 1, padding=1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(8, 2, 3, 1, padding=1),
            )

            self.channel_transformer_conv = nn.ModuleList([nn.Sequential(
                nn.Conv2d(self.transform_hidden + 1, 16, 1, 1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.Conv2d(16, 8, 1, 1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.Conv2d(8, 4, 1, 1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.Conv2d(4, 1, 1, 1, bias=False),
                nn.Tanh()

            ) for _ in range(3)])  # 3 here is a channel size

        elif type == 'custom_gray_conv' :

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(hidden_size, 128, 4, 1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.ConvTranspose2d(128, 64, 5, 2, padding=1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.ConvTranspose2d(64, 32, 4, 2, padding=1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.ConvTranspose2d(32, 32, 3, 2, padding=1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.ConvTranspose2d(32, 16, 3, 2, padding=1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Conv2d(16, 16, 4, 1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Conv2d(16, 1, 3, 1, bias=False),
                nn.Tanh()
            )

            self.transformer = nn.Sequential(
                nn.ConvTranspose2d(self.hidden_size + self.transform_hidden, 128, 4, 1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(64, 32, 4, 2, padding=1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(32, 16, 4, 2, padding=1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(16, 2, 3, 1, padding=1),
            )

            self.channel_transformer_conv = nn.ModuleList([nn.Sequential(
                nn.Conv2d(self.transform_hidden + 1, 64, 1, 1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.Conv2d(64, 32, 1, 1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.Conv2d(32, 16, 1, 1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.Conv2d(16, 1, 1, 1, bias=False),
                nn.Tanh()

            ) for _ in range(3)])  # 3 here is a channel size

        elif type == 'cifar_conv':
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(self.hidden_size, 128, 4, 1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(64, 32, 4, 2, padding=1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(32, 3, 4, 2, padding=1),
                nn.Tanh()
            )

            self.transformer =nn.Sequential(
                nn.ConvTranspose2d(hidden_size + self.transform_hidden, 128, 4, 1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(64, 32, 4, 1, padding=1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(32, 16, 4, 1, padding=1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(16, 2, 3, 1, padding=1, bias=False),

            )

            self.channel_transformer_conv = nn.ModuleList([nn.Sequential(
                nn.Conv2d(self.transform_hidden + 3, 128, 1, 1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.Conv2d(128, 64, 1, 1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.Conv2d(64, 32, 1, 1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.Conv2d(32, 16, 1, 1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.Conv2d(16, 3, 1, 1, bias=False),
                nn.Tanh()

            ) for _ in range(3)])  # 3 here is a channel size

        elif type == 'custom_rgb_conv' :
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(hidden_size, 128, 4, 1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.ConvTranspose2d(64, 32, 4, 2, padding=1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.ConvTranspose2d(32, 32, 4, 2, padding=1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.ConvTranspose2d(32, 3, 4, 2, padding=1, bias=False),
                nn.Tanh()
            )

            self.transformer = nn.Sequential(
                nn.ConvTranspose2d(self.hidden_size + self.transform_hidden, 64, 4, 1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(64, 32, 4, 2, padding=1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(32, 16, 4, 2, padding=1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(16, 2, 4, 2, padding=1),

            )

            self.channel_transformer_conv = nn.ModuleList([nn.Sequential(
                nn.Conv2d(self.transform_hidden + 3, 64, 1, 1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.Conv2d(64, 32, 1, 1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.Conv2d(32, 16, 1, 1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.Conv2d(16, 3, 1, 1, bias=False),
                nn.Tanh()

            ) for _ in range(3)])  # 3 here is a channel size

        elif type == "custom_heavy_rgb_conv" :
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(hidden_size, 128, 4, 1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.ConvTranspose2d(128, 64, 5, 2, padding=1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.ConvTranspose2d(64, 32, 4, 2, padding=1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.ConvTranspose2d(32, 32, 3, 2, padding=1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.ConvTranspose2d(32, 16, 3, 2, padding=1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Conv2d(16, 16, 4, 1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Conv2d(16, 3, 3, 1, bias=False),
                nn.Tanh()
            )

            self.transformer = nn.Sequential(
                nn.ConvTranspose2d(self.hidden_size + self.transform_hidden, 128, 4, 1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(64, 32, 4, 2, padding=1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(32, 16, 4, 2, padding=1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.ConvTranspose2d(16, 2, 3, 1, padding=1),
            )

            self.channel_transformer_conv = nn.ModuleList([nn.Sequential(
                nn.Conv2d(self.transform_hidden + 3, 64, 1, 1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.Conv2d(64, 32, 1, 1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.Conv2d(32, 16, 1, 1, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.Conv2d(16, 3, 1, 1, bias=False),
                nn.Tanh()

            ) for _ in range(3)])  # 3 here is a channel size



        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)

            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal(m.weight.data)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, 2c, h, w) -> (b*c, h, w, 2)"""
        x = x.permute(0, 2, 3, 1);
        # x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), 2)
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, c, h, w) -> (b*c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))
        return x

    @staticmethod
    def _to_b_c_h_w(x, x_shape):
        """(b*c, h, w) -> (b, c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))
        return x

    @staticmethod
    def _get_grid(self, x):
        batch_size, input_height, input_width = x.size(0), x.size(2), x.size(3)
        dtype, cuda = x.data.type(), x.data.is_cuda
        if self._grid_param == (batch_size, input_height, input_width, dtype, cuda):
            return self._grid
        self._grid_param = (batch_size, input_height, input_width, dtype, cuda)
        self._grid = th_generate_grid(batch_size, input_height, input_width, dtype, cuda)
        return self._grid



    def generate_with_interpolation(self, inputs, rel_rec, rel_send, transform_parameter_list, iter_num, sequence_num) :


        output_list = list();
        spatial_transformed_list = list();

        pixel_offset_list = list();

        for transform_parameter in transform_parameter_list :

            output = self.forward(inputs, rel_rec, rel_send, transform_parameter, iter_num, sequence_num);
            output_list.append(output[0]);
            spatial_transformed_list.append(output[2])
            pixel_offset_list.append(output[3])
        return torch.stack(output_list,1), torch.stack(spatial_transformed_list,1), torch.stack(pixel_offset_list,1);


    def spatial_transform(self, input, transform_vector, transform_target):

        transform_cat = torch.cat([input, transform_vector], dim=1).unsqueeze(-1).unsqueeze(-1);


        transformer_result = self.transformer(transform_cat);
        transformer_result = F.interpolate(transformer_result, size=transform_target.shape[2],
                                           mode='bilinear', align_corners=True)  # *inputs[:,0,:];

        offset = transformer_result.permute(0, 2, 3, 1);

        #offset = offset*((torch.abs(offset)>1).cpu().type(torch.FloatTensor).cuda()*1.3);
        # x_offset: (b, c, h, w)
        x_offset, grid = th_batch_map_offsets(transform_target, offset, grid=self._get_grid(self, transform_target))

        final_transformed_result = self._to_b_c_h_w(x_offset, transform_target.shape)

        return final_transformed_result, offset


    def value_transform(self, input, transform_vector, target):
        if 'conv' in self.type :
            transform_matrix_color = torch.cat([input,
                                                transform_vector.unsqueeze(-1).unsqueeze(-1).repeat(1, 1,
                                                                                                             input.shape[
                                                                                                                 2],
                                                                                                             input.shape[
                                                                                                                 3])],
                                               dim=1)
            value_offset = self.channel_transformer_conv[0](transform_matrix_color);

            return target + 0.5*value_offset, value_offset;
        elif 'fc' in self.type :
            #transform_matrix_color = torch.cat([input.unsqueeze(1),
            #                                    transform_vector.unsqueeze(-1).repeat(1,1,input.shape[1])],
            #                                   dim=1)


            transform_matrix_color = torch.cat([input.unsqueeze(-1).unsqueeze(-1),
                                                transform_vector.unsqueeze(-1).unsqueeze(-1)],dim=1);

            '''
            color_transform_list = list();
            for i in range(input.shape[1]) :
                color_transformed_result = self.channel_transformer_conv[i](transform_matrix_color[:,:,i]).squeeze();
                color_transform_list.append(color_transformed_result);
            '''

            value_offset = self.channel_transformer_conv[0](transform_matrix_color).squeeze();

            #return value_offset;
            return target + 0.5*value_offset, value_offset;
            #return torch.stack(color_transform_list, 1);



    def forward(self, inputs, rel_rec, rel_send, transform_vector, iter_num, sequence_num):

        if 'conv' in self.type:
            x = self.mlp1(inputs);
            embed_x = x.view(x.shape[0], -1);

        decode_list = list();
        pixel_offset_list = list();
        value_offset_list = list();

        if 'conv' in self.type:
            decoder_result = self.decoder(embed_x.unsqueeze(-1).unsqueeze(-1));
        elif 'fc' in self.type:

            decoder_result = self.decoder(embed_x);

        init_decode_result = decoder_result;

        for i in range(sequence_num):

            value_transformed_result, value_offset = self.value_transform(init_decode_result,
                                                            transform_vector[:, i, :],
                                                            init_decode_result);

            value_offset_list.append(value_offset);

            if 'conv' in self.type :
                final_transformed_result, pixel_offset = self.spatial_transform(embed_x,
                                                                transform_vector[:, i, :],
                                                                value_transformed_result);
            elif 'fc' in self.type :
                final_transformed_result = value_transformed_result;
                pixel_offset = value_transformed_result;

            decode_list.append(final_transformed_result);

            pixel_offset_list.append(pixel_offset);

        return torch.stack(decode_list, 1), init_decode_result, torch.stack(value_offset_list,
                                                                                          1), torch.stack(pixel_offset_list,
                                                                                                          1);

class Encoder_graph(nn.Module):

    def __init__(self, input_size, hidden_size, transform_hidden, sequence_len,  type='fc'):
        super(Encoder_graph, self).__init__()
        self.hidden_size = hidden_size;
        self.transform_hidden = transform_hidden;
        self.sequence_len = sequence_len;   # l for
        self.type = type;

        if type == 'mnist_conv':
            self.mlp1 = nn.Sequential(
                nn.Conv2d(1, 16, 4, 2, 0),
                nn.LeakyReLU(),
                nn.Conv2d(16, 8, 4, 2, 0),
                nn.LeakyReLU(),
                nn.Conv2d(8, 4, 3, 1, 0),
                nn.LeakyReLU(),
            )

        elif type == 'omniglot_conv':
            self.mlp1 = nn.Sequential(
                nn.Conv2d(1, 32, 4, 2, 0),
                nn.LeakyReLU(),
                nn.Conv2d(32, 16, 4, 2, 0),
                nn.LeakyReLU(),
                nn.Conv2d(16, 8, 3, 1, 0),
                nn.LeakyReLU(),
            )

        elif type == 'fmnist_conv':
            self.mlp1 = nn.Sequential(
                nn.Conv2d(1, 32, 4, 2, 0),
                nn.LeakyReLU(),
                nn.Conv2d(32, 16, 4, 2, 0),
                nn.LeakyReLU(),
                nn.Conv2d(16, 8, 3, 1, 0),
                nn.LeakyReLU(),
            )

        elif type == 'custom_gray_conv':
            self.mlp1 = nn.Sequential(
                nn.Conv2d(1, 16, 4, 2, 0),
                nn.LeakyReLU(),
                nn.Conv2d(16, 32, 4, 2, 0),
                nn.LeakyReLU(),
                nn.Conv2d(32, 32, 4, 2, 0),
                nn.LeakyReLU(),
                nn.Conv2d(32, 32, 3, 1, 0),
                nn.LeakyReLU(),
            )

        elif type == 'cifar_conv':
            self.mlp1 = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 0),
                nn.LeakyReLU(),
                nn.Conv2d(64, 32, 4, 1, 0),
                nn.LeakyReLU(),
                nn.Conv2d(32, 16, 4, 1, 0),
                nn.LeakyReLU(),
                nn.Conv2d(16, 16, 4, 1, 0),
                nn.LeakyReLU(),
                nn.Conv2d(16, 16, 3, 1, 0),
                nn.LeakyReLU(),
            )

        elif type == "custom_rgb_conv"  or type=="custom_heavy_rgb_conv":
            self.mlp1 = nn.Sequential(
                nn.Conv2d(3, 16, 4, 2, 0),
                nn.LeakyReLU(),
                nn.Conv2d(16, 32, 4, 2, 0),
                nn.LeakyReLU(),
                nn.Conv2d(32, 32, 4, 2, 0),
                nn.LeakyReLU(),
                nn.Conv2d(32, 32, 3, 1, 0),
                nn.LeakyReLU(),
            )



        ### Graph Iteration to encode Inference Adjacency
        self.mlp2 = SLP(hidden_size * 2, hidden_size)
        self.mlp3 = SLP(hidden_size, hidden_size)

        self.mlp4 = SLP(hidden_size * 2 + hidden_size, hidden_size)

        # self.fc_out = nn.Linear(hidden_size, 2);
        self.fc_out_k = nn.ModuleList(
            [nn.Linear(hidden_size * (sequence_len - 1), sequence_len - 1) for _ in range(sequence_len)]);


        self.init_weights()

        ### Graph Iteration to encode Transformation Parameter

        # no initialization of network. (more emphasize on 0 or 1 initialization of alpha)
        self.msg_fc1 = MLP(transform_hidden * 2, transform_hidden, transform_hidden)
        # [nod, recved msg] to [nod embedding]
        self.node_embed = SLP(transform_hidden, transform_hidden);

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal(m.weight.data)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def edge2node(self, x, rel_rec, rel_send):

        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):

        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([receivers, senders], dim=-1)

        return edges / x.shape[1]

    def node2edge_tp(self, x, rel_rec, rel_send):

        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([receivers, senders], dim=-1)
        return edges

    def edge2node_tp(self, x, rel_rec, rel_send):

        # NOTE: Assumes that we have the same graph across all samples.
        # we collect message information
        incoming = torch.matmul(rel_send.permute(0, 2, 1), x)  #######
        #incoming = torch.matmul(rel_rec.permute(0, 2, 1), x)  #######
        return incoming

    def iter_node2node(self, rel_rec, rel_send, rel_type, sequence_num=5, transform_type='color'):

        transform_x = Variable(torch.zeros(rel_type.shape[0], sequence_num, self.transform_hidden));
        x = Variable(torch.zeros(rel_type.shape[0], sequence_num, self.transform_hidden));  # initial transform ==> 0

        all_msgs = Variable(torch.zeros(rel_type.shape[0], sequence_num, self.transform_hidden));

        if rel_type.is_cuda:
            all_msgs = all_msgs.cuda()
            transform_x = transform_x.cuda();
            x = x.cuda();

        x[:, 0, :] = 1.0;  ### initialize source as 1 vector and 0 for targets

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(0, sequence_num):
            x = self.node_embed(x);

            x = self.node2edge_tp(x, rel_type.unsqueeze(-1) * rel_rec, rel_type.unsqueeze(-1) * rel_send);

            msg = self.msg_fc1(x)

            x = self.edge2node_tp(msg, rel_type.unsqueeze(-1) * rel_rec,
                               rel_type.unsqueeze(-1) * rel_send);  ## aggregated msg

            all_msgs += x;

        # Aggregate all msgs to receiver  and 0 for source transform parameter
        transform_x[:, 1:, :] = all_msgs[:, 1:, :];

        return transform_x;

    def generate_with_interpolation(self, inputs, rel_rec, rel_send, rel_full, sequence_num=5, interpol_range=[0.1,1.0], gen_num= 23) :

        start_val = interpol_range[0];
        end_val = interpol_range[1];

        ratios = np.linspace(start_val, end_val, gen_num);
        transform_parameter_list = list();


        for rate in ratios:
            transform_parameter = self.forward(inputs, rel_rec, rel_send, rel_full, sequence_num=sequence_num, rel_rate=rate);
            transform_parameter_list.append(transform_parameter);

        return transform_parameter_list;


    def forward(self, inputs, rel_rec, rel_send, rel_full, sequence_num=5, rel_rate = 1.0):

        x = inputs;

        if 'fc' in self.type:
            x = self.mlp1(x);
        elif 'conv' in self.type:
            encode_list = list();
            for i in range(x.shape[1]):
                encode_list.append(self.mlp1(x[:, i, :, :]));

            x = torch.stack(encode_list, dim=1);
            x = x.view(x.shape[0], x.shape[1], -1);

        x = self.node2edge(x, rel_rec, rel_send);

        x = self.mlp2(x)

        x_skip = x;

        x = self.edge2node(x, rel_rec, rel_send);

        x = self.mlp3(x);

        x = self.node2edge(x, rel_rec, rel_send);

        x = torch.cat([x, x_skip], dim=-1);

        x = self.mlp4(x);
        x = x.view(x.shape[0], inputs.shape[1], -1);

        out_list = list();

        for i in range(x.shape[1]):
            out_k = self.fc_out_k[i](x[:, i, :]);
            out_list.append(out_k);

        out = torch.stack(out_list, 1);
        # out = self.fc_out(x);


        edges = F.softmax(out, -1);

        edges = edges.contiguous().view(edges.size(0), -1) * rel_rate;

        transform_vector = self.iter_node2node(rel_rec, rel_send, edges, sequence_num);

        return transform_vector;

