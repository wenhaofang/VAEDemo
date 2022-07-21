# SAE | the same as AE

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, layers_size, latent_size):
        super(Encoder, self).__init__()
        self.mlps = nn.Sequential()
        for layer_i, (
            i_size,
            o_size,
        ) in enumerate(zip(
            layers_size,
            layers_size[1:] + [latent_size],
        )):
            self.mlps.add_module(
                name = 'e_l_{}'.format(layer_i), module = nn.Linear(i_size, o_size)
            )
            self.mlps.add_module(
                name = 'e_a_{}'.format(layer_i), module = nn.ReLU()
            )

    def forward(self, x):
        '''
        Params:
            x: Torch Tensor (batch_size, hidden_size) # hidden_size == layers_size[0]
        Return:
            z: Troch Tensor (batch_size, latent_size)
        '''
        z = self.mlps(x)

        return z

class Decoder(nn.Module):
    def __init__(self, latent_size, layers_size):
        super(Decoder, self).__init__()
        self.mlps = nn.Sequential()
        for layer_i, (
            i_size,
            o_size,
        ) in enumerate(zip(
            [latent_size] + layers_size[:-1],
            (layers_size)
        )):
            self.mlps.add_module(
                name = 'd_l_{}'.format(layer_i), module = nn.Linear(i_size, o_size)
            )
            self.mlps.add_module(
                name = 'd_a_{}'.format(layer_i), module = nn.ReLU() if layer_i + 1 < len(layers_size) else nn.Sigmoid()
            )

    def forward(self, z):
        '''
        Params:
            z: Torch Tensor (batch_size, latent_size)
        Return:
            x: Torch Tensor (batch_size, output_size) # output_size == layers_size[-1]
        '''
        x = self.mlps(z)

        return x

class SAE(nn.Module):
    def __init__(self, encoder_layers_size, latent_size, decoder_layers_size):
        super(SAE, self).__init__()

        assert type(latent_size) == int
        assert type(encoder_layers_size) == list
        assert type(decoder_layers_size) == list

        self.latent_size = latent_size
        self.encoder = Encoder(encoder_layers_size, latent_size)
        self.decoder = Decoder(latent_size, decoder_layers_size)

    def forward(self, x):
        '''
        Params:
            x: Torch Tensor (batch_size, hidden_size)
        Return:
            z: Torch Tensor (batch_size, latent_size)
            y: Torch Tensor (batch_size, output_size) # output_size == hidden_size
        '''
        z = self.encoder(x)

        y = self.decoder(z)

        return y, z

def get_module(option):
    return SAE(
        option.encoder_layers_size, option.latent_size, option.decoder_layers_size
    )

if  __name__ == '__main__':

    from utils.parser import get_parser

    parser = get_parser()
    option = parser.parse_args()

    module = get_module(option)

    channel = 1
    image_w = 28
    image_h = 28

    x = torch.randn((option.batch_size, channel * image_w * image_h))

    reco_x, z = module(x)

    print(reco_x.shape) # (batch_size, channel * image_w * image_h)
    print(z.shape)      # (batch_size, latent_size)
