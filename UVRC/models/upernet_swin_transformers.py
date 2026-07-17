from functools import partial

import torch
import torch.nn as nn
from transformers import UperNetForSemanticSegmentation

from models.common import STFT, get_act, get_norm, Upscale, Downscale, cac2cws, cws2cac


class TFC_TDF(nn.Module):
    def __init__(self, in_c, c, l, f, bn, norm, act):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(l):
            block = nn.Module()

            block.tfc1 = nn.Sequential(
                norm(in_c),
                act,
                nn.Conv2d(in_c, c, 3, 1, 1, bias=False),
            )
            block.tdf = nn.Sequential(
                norm(c),
                act,
                nn.Linear(f, f // bn, bias=False),
                norm(c),
                act,
                nn.Linear(f // bn, f, bias=False),
            )
            block.tfc2 = nn.Sequential(
                norm(c),
                act,
                nn.Conv2d(c, c, 3, 1, 1, bias=False),
            )
            block.shortcut = nn.Conv2d(in_c, c, 1, 1, 0, bias=False)

            self.blocks.append(block)
            in_c = c

    def forward(self, x):
        for block in self.blocks:
            s = block.shortcut(x)
            x = block.tfc1(x)
            x = x + block.tdf(x)
            x = block.tfc2(x)
            x = x + s
        return x


class Swin_UperNet_Model(nn.Module):
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

        self.swin_upernet_model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-swin-large")

        self.swin_upernet_model.auxiliary_head.classifier = nn.Conv2d(256, c, kernel_size=(1, 1), stride=(1, 1))
        self.swin_upernet_model.decode_head.classifier = nn.Conv2d(512, c, kernel_size=(1, 1), stride=(1, 1))
        self.swin_upernet_model.backbone.embeddings.patch_embeddings.projection = nn.Conv2d(c, 192, kernel_size=(4, 4), stride=(4, 4))

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

        x = self.swin_upernet_model(x).logits

        x = x.transpose(-1, -2)

        x = x * first_conv_out  # reduce artifacts

        x = self.final_conv(torch.cat([mix, x], 1))

        x = self.cws2cac(x)

        if self.num_target_instruments > 1:
            b, c, f, t = x.shape
            x = x.reshape(b, self.num_target_instruments, -1, f, t)

        x = self.stft.inverse(x)
        return x


if __name__ == "__main__":
    model = UperNetForSemanticSegmentation.from_pretrained("./results/", ignore_mismatched_sizes=True)
    print(model)
    print(model.auxiliary_head.classifier)
    print(model.decode_head.classifier)

    x = torch.zeros((2, 16, 512, 512), dtype=torch.float32)
    res = model(x)
    print(res.logits.shape)
    model.save_pretrained('./results/')
