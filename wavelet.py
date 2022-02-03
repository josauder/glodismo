import pywt
import torch
from conf import device

class WT(object):
    """Adapted from: https://lernapparat.de/2d-wavelet-transform-pytorch"""
    def __init__(self):
        w = pywt.Wavelet('bior2.2')
        dec_hi = torch.Tensor(w.dec_hi[::-1]).to(device)
        dec_lo = torch.Tensor(w.dec_lo[::-1]).to(device)
        rec_hi = torch.Tensor(w.rec_hi).to(device)
        rec_lo = torch.Tensor(w.rec_lo).to(device)

        self.filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

        self.inv_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                                   rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                                   rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                                   rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    def wt(self, vimg, levels=2):
        h = vimg.size(2)
        w = vimg.size(3)
        padded = torch.nn.functional.pad(vimg, (2, 2, 2, 2), 'constant')

        res = torch.nn.functional.conv2d(padded, self.filters[:, None], stride=2)
        res = res.contiguous()
        if levels > 1:
            res[:, :1] = self.wt(res[:, :1], levels - 1)
        res = res.reshape(-1, 2, h // 2, w // 2).transpose(1, 2)
        res = res.reshape(-1, 1, h, w)
        return res

    def iwt(self, vres, levels=2):
        h = vres.size(2)
        w = vres.size(3)
        res = vres.reshape(-1, h // 2, 2, w // 2).transpose(1, 2).contiguous().reshape(-1, 4, h // 2, w // 2).clone()
        if levels > 1:
            res[:, :1] = self.iwt(res[:, :1], levels=levels - 1)
        res = torch.nn.functional.conv_transpose2d(res, self.inv_filters[:, None], stride=2)
        res = res[:, :, 2:-2, 2:-2]
        return res

    