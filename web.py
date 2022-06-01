#!/usr/bin/env python3
import datetime
import imghdr
import json
import os
import uuid
from multiprocessing import Lock, Pool

import numpy as np
from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    send_from_directory,
)

from sr import SR

app = Flask(__name__)

# constrain max 512MB
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 512
app.config['DATA_PATH'] = './data'


def ValImg(stream):
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)
    if not format:
        return None
    return format if format != 'jpeg' else 'jpg'


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    task_handle = uuid.uuid4().hex
    exts = ['jpg', 'png']
    ret = {'Status': 0, 'StatusText': '', 'Count': 0, 'Handle': task_handle}
    data_path = os.path.join(app.config['DATA_PATH'], task_handle)
    img_path = os.path.join(data_path, 'upload')
    os.makedirs(data_path)
    os.makedirs(img_path)
    opt = request.form
    print(opt)
    count = 0
    filename = []
    for file in request.files.getlist('pic'):
        ext = file.filename.split('.')[-1]
        if ext not in exts or ValImg(file.stream) != ext:
            continue
        file.save(os.path.join(data_path, 'upload', file.filename))
        filename.append(file.filename)
        count += 1
    info = {
        'Handle': task_handle,
        'Time': str(datetime.datetime.now()),
        'Count': count,
        'file': filename,
        'opt': {
            'enhancement': opt.get('enhancement'),
            'ensemble': opt.get('ensemble'),
            'gan': opt.get('gan'),
            'scale': int(opt.get('scale')),
        },
    }
    print(info)
    with open(
        os.path.join(app.config['DATA_PATH'], task_handle, 'info.json'), 'w'
    ) as f:
        json.dump(info, f)
    status = {
        'Count': count,
        'Handle': task_handle,
        'Process': 'Pending',
        'FinishImg': 0,
        'FileName': '',
        'Download': '#',
    }
    with open(
        os.path.join(app.config['DATA_PATH'], task_handle, 'status.json'), 'w'
    ) as f:
        json.dump(status, f)

    ret['Count'] = count
    if count == 0:
        ret['Status'] = 701
        ret['StatusText'] = 'No images available'
        return jsonify(ret)

    f = SR(task_handle, app.config['DATA_PATH'])
    pool.apply_async(f)
    # f()
    ret = jsonify(ret)
    ret.set_cookie('TASKHANDLE', task_handle)
    return ret


@app.route('/status', methods=['GET'])
def status():
    ret = {
        'Status': 0,
        'StatusText': '',
        'Count': 0,
        'Handle': '',
        'Process': '',
        'FinishImg': 0,
        'FileName': '',
        'Download': '#',
    }

    ret['Handle'] = request.args.get('handle') or request.cookies.get('TASKHANDLE')
    if not ret['Handle']:
        ret['Status'] = 702
        ret['StatusText'] = 'Missing parameters.'
        return ret

    st_path = os.path.join(app.config['DATA_PATH'], ret['Handle'], 'status.json')
    if os.path.exists(st_path):
        with open(st_path, 'r') as f:
            status = json.load(f)
        ret['Count'] = status['Count']
        ret['FinishImg'] = status['FinishImg']
        ret['Process'] = status['Process']
        ret['Download'] = status['Download']
        if ret['Process'] == "Failed":
            ret['Status'] = 704
            ret['StatusText'] = 'Task failed.'
    else:
        ret['Status'] = 703
        ret['StatusText'] = 'Task not found.'
    return jsonify(ret)


@app.route('/download/<handle>/<filename>')
def download(handle, filename):
    path = os.path.join(app.config['DATA_PATH'], handle, 'result')
    return send_from_directory(path, filename, as_attachment=True)


@app.route('/favicon.svg')
def favicon():
    return app.send_static_file('img/favicon.svg')


if __name__ == '__main__':
    global pool
    pool = Pool(processes=1)
    os.makedirs(app.config['DATA_PATH'], exist_ok=True)
    app.run(host='0.0.0.0', debug=False, port=8888, threaded=True)
    pool.close()
    pool.join()
