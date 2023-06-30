import flow_vis
import numpy as np
from imageio.v3 import imread, imwrite



def read(file):
    if file.endswith('.flo'):
        return readFlow(file)
    elif file.endswith('ppm') or file.endswith('png'):
        return readPPM(file)
    else:
        raise Exception('Invalid Filetype {}', file)


def readFlow(file: str) -> np.ndarray:
    f = open(file, 'rb')
    header = np.fromfile(f, np.float32, count=1).squeeze()
    if header != 202021.25:
        raise Exception('Invalid .flo file {}', file)
    w = np.fromfile(f, np.int32, 1).squeeze()
    h = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, w*h*2).reshape((h, w, 2))

    return flow

def readPPM(file: str) -> np.ndarray:
    return imread(file)

def write_flow(filename: str, flow:np.array):

    flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=False)
    imwrite(filename, flow_color)

def write_flo_file(filename: str, flow):
    f = open(filename, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)