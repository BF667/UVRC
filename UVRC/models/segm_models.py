import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from models.common import STFT, get_act, cac2cws, cws2cac


# Mapping from decoder_type string to smp class
_DECODER_MAP = {
    "unet": smp.Unet,
    "fpn": smp.FPN,
    "unet++": smp.UnetPlusPlus,
    "manet": smp.MAnet,
    "linknet": smp.Linknet,
    "pspnet": smp.PSPNet,
    "pan": smp.PAN,
    "deeplabv3": smp.DeepLabV3,
    "deeplabv3plus": smp.DeepLabV3Plus,
}

# Mapping from decoder_type to config key holding extra options
_DECODER_CONFIG_KEYS = {
    "unet": "decoder_unet",
    "fpn": "decoder_fpn",
    "unet++": "decoder_unet_plus_plus",
    "manet": "decoder_manet",
    "linknet": "decoder_linknet",
    "pspnet": "decoder_pspnet",
    "pan": "decoder_pan",
    "deeplabv3": "decoder_deeplabv3",
    "deeplabv3plus": "decoder_deeplabv3plus",
}


def get_decoder(config, c):
    """Instantiate a segmentation model decoder from config."""
    decoder_type = config.model.decoder_type
    if decoder_type not in _DECODER_MAP:
        raise ValueError(f"Unknown decoder type: {decoder_type}")

    decoder_cls = _DECODER_MAP[decoder_type]
    decoder_options = {}

    config_key = _DECODER_CONFIG_KEYS.get(decoder_type)
    if config_key:
        try:
            decoder_options = dict(getattr(config, config_key))
        except AttributeError:
            pass

    return decoder_cls(
        encoder_name=config.model.encoder_name,
        encoder_weights="imagenet",
        in_channels=c,
        classes=c,
        **decoder_options,
    )


class Segm_Models_Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        act = get_act(act_type=config.model.act)

        self.num_target_instruments = 1 if config.training.target_instrument else len(config.training.instruments)
        self.num_subbands = config.model.num_subbands

        dim_c = self.num_subbands * config.audio.num_channels * 2
        c = config.model.num_channels
        f = config.audio.dim_f // self.num_subbands

        self.first_conv = nn.Conv2d(dim_c, c, 1, 1, 0, bias=False)

        self.unet_model = get_decoder(config, c)

        self.final_conv = nn.Sequential(
            nn.Conv2d(c + dim_c, c, 1, 1, 0, bias=False),
            act,
            nn.Conv2d(c, self.num_target_instruments * dim_c, 1, 1, 0, bias=False)
        )

        self.stft = STFT(config.audio)

    def cac2cws(self, x):
        return cac2cws(x, self.num_subbands)

    def cws2cac(self, x):
        return cws2cac(x, self.num_subbands)

    def forward(self, x):

        x = self.stft(x)

        mix = x = self.cac2cws(x)

        first_conv_out = x = self.first_conv(x)

        x = x.transpose(-1, -2)

        x = self.unet_model(x)

        x = x.transpose(-1, -2)

        x = x * first_conv_out  # reduce artifacts

        x = self.final_conv(torch.cat([mix, x], 1))

        x = self.cws2cac(x)

        if self.num_target_instruments > 1:
            b, c, f, t = x.shape
            x = x.reshape(b, self.num_target_instruments, -1, f, t)

        x = self.stft.inverse(x)
        return x
