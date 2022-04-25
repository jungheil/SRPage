import json
import os
import shutil
import zipfile

from atomicwrites import atomic_write


class Process:
    def __init__(self, handle, data_path='./data', tmp_path='./tmp'):
        self.handle = handle
        self.data_path = data_path
        self.tmp_path = './tmp'
        self.status = {
            'Count': 0,
            'Handle': self.handle,
            'Process': 'Pending',
            'FinishImg': 0,
            'FileName': '',
            'Download': '#',
        }
        os.makedirs(tmp_path, exist_ok=True)

    def __call__(self):
        info = self._LoadInfo()
        self.status['Count'] = info['Count']
        os.makedirs(os.path.join(self.tmp_path, self.handle), exist_ok=True)
        self._Process()
        self._Pac()
        print('Process Done! handle:{}'.format(self.handle))

    def _Process(self):
        pass

    def _WriteStatus(self):
        path = os.path.join(self.data_path, self.handle, 'status.json')
        with atomic_write(path, overwrite=True) as f:
            json.dump(self.status, f)

    def _Pac(self):
        self.status['Process'] = 'Packing'
        self._WriteStatus()
        files = os.listdir(os.path.join(self.tmp_path, self.handle))
        out_path = os.path.join(self.data_path, self.handle, 'result')
        os.makedirs(out_path, exist_ok=True)
        if len(files) > 1:
            self.status['FileName'] = '{}.zip'.format(self.handle)
            z = zipfile.ZipFile(
                os.path.join(out_path, self.status['FileName']),
                'w',
                zipfile.ZIP_DEFLATED,
            )
            for f in files:
                z.write(os.path.join(self.tmp_path, self.handle, f), f)
            z.close()
        elif len(files) == 1:
            self.status['FileName'] = files[0]
            shutil.copyfile(
                os.path.join(self.tmp_path, self.handle, files[0]),
                os.path.join(out_path, files[0]),
            )
        shutil.rmtree(os.path.join(self.tmp_path, self.handle), ignore_errors=True)
        self.status['Process'] = 'Done'
        self.status['Download'] = '/download/{}/{}'.format(
            self.handle, self.status['FileName']
        )
        self._WriteStatus()

    def _LoadInfo(self):
        with open(os.path.join(self.data_path, self.handle, 'info.json'), 'r') as f:
            info = json.load(f)
        return info


class Simulator(Process):
    def __init__(self, handle, data_path, tmp_path='./tmp'):
        super().__init__(handle, data_path, tmp_path)

    def __call__(self):
        return super().__call__()

    def _Process(self):
        self.status['Process'] = 'Working'
        files = os.listdir(os.path.join(self.data_path, self.handle, 'upload'))
        for i, f in enumerate(files):
            shutil.copyfile(
                os.path.join(self.data_path, self.handle, 'upload', f),
                os.path.join(self.tmp_path, self.handle, f),
            )
            self.status['FinishImg'] = i + 1
            self._WriteStatus()
