import glob
import os

import imageio
import torch
import yaml
from basicsr.models import build_model
from basicsr.utils.options import ordered_yaml

from process import Process


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
        img = torch.from_numpy(img).to(self.device).permute(2, 0, 1) / 255.0
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

    def _LoadModel(self, info):
        with open(
            '/home/li/Git/BasicSR/options/test/TZ/test_TZ_x2_UW.yml', mode='r'
        ) as f:
            self.opt = yaml.load(f, Loader=ordered_yaml()[0])
        self.opt['is_train'] = False
        self.opt['dist'] = False
        self.model = build_model(self.opt)

    def _Process(self):
        self.status['Process'] = 'Working'

        try:
            self._LoadModel(self.info)
            self.device = torch.device('cuda' if self.opt['num_gpu'] != 0 else 'cpu')
            data = PredData(
                os.path.join(self.data_path, self.handle, 'upload'), self.device
            )

            for i, img in enumerate(data):
                out = self.model.net_g(img) * 255.0
                # out = out.type(torch.uint8)
                out = out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                torch.cuda.empty_cache()
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
