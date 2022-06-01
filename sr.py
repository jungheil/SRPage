import glob
import os
from turtle import forward

import imageio
import numpy as np
import torch
import yaml
from basicsr.models import build_model
from basicsr.utils.options import ordered_yaml

from enhancement import HazeRemoval
from process import Process


def add_clear(fun):
    def warp(*args, **kwargs):
        ret = fun(*args, **kwargs)
        torch.cuda.empty_cache()
        return ret

    return warp


class PredData(torch.utils.data.Dataset):
    def __init__(self, path, device):
        self.path = path
        self.ext = ['jpg', 'png', 'bmp']
        self.imgs = []
        self.device = device
        self._scan()

    def _scan(self):
        self.imgs = []
        for e in self.ext:
            self.imgs.extend(sorted(glob.glob(os.path.join(self.path, '*.' + e))))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = imageio.imread(self.imgs[i])
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        img = (
            torch.from_numpy(img).to(self.device).permute(2, 0, 1).unsqueeze(0) / 255.0
        )
        return img

    def get_img_name(self, i):
        return os.path.basename(self.imgs[i])


class SR(Process):
    def __init__(self, handle, data_path, tmp_path='./tmp'):
        super().__init__(handle, data_path, tmp_path)

    def __call__(self):
        return super().__call__()

    def __del__(self):
        torch.cuda.empty_cache()

    def _LoadModel(self):
        if self.info['opt']['scale'] == 4:
            if self.info['opt']['gan']:
                with open('arch/FRSN_x4_UW_GAN.yml', mode='r') as f:
                    self.opt = yaml.load(f, Loader=ordered_yaml()[0])
            else:
                with open('arch/FRSN_x4_UW.yml', mode='r') as f:
                    self.opt = yaml.load(f, Loader=ordered_yaml()[0])
        else:
            if self.info['opt']['gan']:
                with open('arch/FRSN_x2_UW_GAN.yml', mode='r') as f:
                    self.opt = yaml.load(f, Loader=ordered_yaml()[0])
            else:
                with open('arch/FRSN_x2_UW.yml', mode='r') as f:
                    self.opt = yaml.load(f, Loader=ordered_yaml()[0])

        self.opt['is_train'] = False
        self.opt['dist'] = False
        self.model = build_model(self.opt)

    def _Process(self):
        self.status['Process'] = 'Working'

        if self.info['opt']['enhancement']:
            hr = HazeRemoval()

        # self._LoadModel()
        # self.device = torch.device('cuda' if self.opt['num_gpu'] != 0 else 'cpu')
        # data = PredData(
        #     os.path.join(self.data_path, self.handle, 'upload'), self.device
        # )

        # for i, img in enumerate(data):
        #     if self.info['opt']['ensemble']:
        #         out = (
        #             self.forward_x8(img, forward_function=self.model.net_g) * 255
        #         )
        #     else:
        #         with torch.no_grad():
        #             out = self.model.net_g(img) * 255.0
        #     # out = out.type(torch.uint8)
        #     out = out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        #     torch.cuda.empty_cache()
        #     if self.info['opt']['enhancement']:
        #         out = hr.get(out)
        #     imageio.imwrite(
        #         os.path.join(self.tmp_path, self.handle, data.get_img_name(i)), out
        #     )

        #     self.status['FinishImg'] = i + 1
        #     self._WriteStatus()

        try:
            self._LoadModel()
            self.device = torch.device('cuda' if self.opt['num_gpu'] != 0 else 'cpu')
            data = PredData(
                os.path.join(self.data_path, self.handle, 'upload'), self.device
            )

            for i, img in enumerate(data):
                if self.info['opt']['ensemble']:
                    out = self.forward_x8(img, forward_function=self.model.net_g) * 255
                else:
                    with torch.no_grad():
                        out = self.model.net_g(img) * 255.0
                # out = out.type(torch.uint8)
                out = out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                torch.cuda.empty_cache()
                if self.info['opt']['enhancement']:
                    out = hr.get(out)
                imageio.imwrite(
                    os.path.join(self.tmp_path, self.handle, data.get_img_name(i)), out
                )

                self.status['FinishImg'] = i + 1
                self._WriteStatus()

        except Exception:
            self.status['Process'] = 'Failed'
            self._WriteStatus()

        finally:
            torch.cuda.empty_cache()

    def forward_x8(self, *args, forward_function=None):
        def _transform(v, op):
            v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)

            return ret

        list_x = []
        for a in args:
            x = [a]
            for tf in 'v', 'h', 't':
                x.extend([_transform(_x, tf) for _x in x])

            list_x.append(x)

        list_y = []
        for x in zip(*list_x):
            with torch.no_grad():
                y = forward_function(*x)
            if not isinstance(y, list):
                y = [y]
            if not list_y:
                list_y = [[_y] for _y in y]
            else:
                for _list_y, _y in zip(list_y, y):
                    _list_y.append(_y)

        for _list_y in list_y:
            for i in range(len(_list_y)):
                if i > 3:
                    _list_y[i] = _transform(_list_y[i], 't')
                if i % 4 > 1:
                    _list_y[i] = _transform(_list_y[i], 'h')
                if (i % 4) % 2 == 1:
                    _list_y[i] = _transform(_list_y[i], 'v')

        y = [torch.cat(_y, dim=0).mean(dim=0, keepdim=True) for _y in list_y]
        if len(y) == 1:
            y = y[0]

        return y
