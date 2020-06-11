import torch.nn as nn
from utils import *
from torch_deform_conv.deform_conv import th_batch_map_offsets, th_generate_grid


_EPS = 1e-10


class ResidualBlock(nn.Module) :
    ''' Residual Block '''
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            )

    def forward(self, x):
        return x + self.main(x)

class SLP(nn.Module) :
    """One-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_out, do_prob=0., activation = 'elu'):
        super(SLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_out)
        self.bn1 = nn.BatchNorm1d(n_out);
        self.dropout_prob = do_prob
        self.activation = activation;

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, inputs, do_on = None):

        if do_on is None :
            do_on = self.training;

        # Input shape: [num_sims, num_things, num_features]
        if self.activation == 'elu' :
            x = F.elu(self.fc1(inputs));
            x = F.dropout(x, self.dropout_prob, training=do_on)
        elif self.activation =='tanh' :
            x = F.tanh(self.fc1(inputs))
        elif self.activation == 'softmax' :
            x = F.softmax(self.fc1(inputs));

        return x


class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0., bias=True, activation='elu'):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.bn1 = nn.BatchNorm1d(n_hid);
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn2 = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob
        self.bias = bias
        self.activation = activation

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                if self.bias :
                    m.bias.data.fill_(0.1)
                elif not self.bias :
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm1(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn1(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def batch_norm2(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn2(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        #x = F.elu(self.batch_norm1(self.fc1(inputs)))

        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        if self.activation == 'elu':
            x = F.elu(self.fc2(x))
        elif self.activation == 'tanh' :
            x = F.tanh(self.fc2(x))

        return x




