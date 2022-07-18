# CVAE

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, layers_size, latent_size, num_labels):
        super(Encoder, self).__init__()
        self.mlps = nn.Sequential()
        for layer_i, (
            i_size,
            o_size,
        ) in enumerate(zip(
            layers_size[:-1],
            layers_size[+1:],
        )):
            if  layer_i == 0:
                i_size += num_labels # diff

            self.mlps.add_module(
                name = 'e_l_{}'.format(layer_i), module = nn.Linear(i_size, o_size)
            )
            self.mlps.add_module(
                name = 'e_a_{}'.format(layer_i), module = nn.ReLU()
            )

        self.mean_fc = nn.Linear(layers_size[-1], latent_size)
        self.lstd_fc = nn.Linear(layers_size[-1], latent_size)

    def forward(self, x, c):
        '''
        Params:
            x   : Torch Tensor (batch_size, hidden_size)
            c   : Torch Tensor (batch_size, onehot_size)
        Return:
            mean: Troch Tensor (batch_size, latent_size)
            lstd: Troch Tensor (batch_size, latent_size)
        '''
        x = torch.cat((x, c), dim = -1) # diff

        x = self.mlps(x)

        mean = self.mean_fc(x)
        lstd = self.lstd_fc(x)

        return mean, lstd

class Decoder(nn.Module):
    def __init__(self, latent_size, layers_size, num_labels):
        super(Decoder, self).__init__()
        self.mlps = nn.Sequential()
        for layer_i, (
            i_size,
            o_size,
        ) in enumerate(zip(
            [latent_size + num_labels] + layers_size[:-1], # diff
            (layers_size)
        )):
            self.mlps.add_module(
                name = 'd_l_{}'.format(layer_i), module = nn.Linear(i_size, o_size)
            )
            self.mlps.add_module(
                name = 'd_a_{}'.format(layer_i), module = nn.ReLU() if layer_i + 1 < len(layers_size) else nn.Sigmoid()
            )

    def forward(self, z, c):
        '''
        Params:
            z: Torch Tensor (batch_size, latent_size)
            c: Torch Tensor (batch_size, onehot_size)
        Return:
            x: Torch Tensor (batch_size, output_size)
        '''
        z = torch.cat((z, c), dim = -1) # diff

        x = self.mlps(z)

        return x

class CVAE(nn.Module):
    def __init__(self, encoder_layers_size, latent_size, decoder_layers_size, num_labels):
        super(CVAE, self).__init__()

        assert type(latent_size) == int
        assert type(encoder_layers_size) == list
        assert type(decoder_layers_size) == list

        self.latent_size = latent_size
        self.encoder = Encoder(encoder_layers_size, latent_size, num_labels)
        self.decoder = Decoder(latent_size, decoder_layers_size, num_labels)

    def forward(self, x, c):
        '''
        Params:
            x   : Torch Tensor (batch_size, hidden_size)
            c   : Torch Tensor (batch_size, onehot_size)
        Return:
            mean: Troch Tensor (batch_size, latent_size)
            lstd: Troch Tensor (batch_size, latent_size)
            z   : Torch Tensor (batch_size, latent_size)
            y   : Torch Tensor (batch_size, output_size) # output_size == hidden_size
        '''
        mean, lstd = self.encoder(x, c) # diff

        z = self.reparameterize(mean, lstd)

        reco_x = self.decoder(z, c) # diff

        return reco_x, z, mean, lstd

    def predict(self, c):
        '''
        Params:
            c: Torch Tensor (n, onehot_size)
        Return:
            x: Torch Tensor (n, output_size)
        '''
        z = torch.randn((c.shape[0], self.latent_size), device = c.device) # diff

        x = self.decoder(z, c)

        return x

    def reparameterize(self, mean, lstd):
        return mean + torch.randn_like(lstd) * torch.exp(lstd) # lstd -> torch.log(std)

def get_module (option):
    return CVAE(
        option.encoder_layers_size, option.latent_size, option.decoder_layers_size, option.num_labels
    )

if  __name__ == '__main__':

    from utils.parser import get_parser

    parser = get_parser()
    option = parser.parse_args()

    module = get_module(option)

    from utils.misc import to_onehot

    channel = 1
    image_w = 28
    image_h = 28

    x = torch.randn(( option.batch_size,  channel * image_w * image_h))
    c = to_onehot(
        torch.randint(option.num_labels, (option.batch_size,)),
        option.num_labels
    )

    reco_x, z, mean, lstd = module(x, c)

    print(reco_x.shape) # (batch_size, channel * image_w * image_h)
    print(z.shape)      # (batch_size, latent_size)
    print(mean.shape)   # (batch_size, latent_size)
    print(lstd.shape)   # (batch_size, latent_size)

    c = torch.arange(option.num_labels)
    c = to_onehot(c, option.num_labels)

    y = module.predict(c)

    print(y.shape)      # (n, channel * image_w * image_h)
