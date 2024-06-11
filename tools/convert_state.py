import torch
import msgpack
import sys

def convert_to_fr_state(input_path, output_path):
    w = torch.load(input_path, map_location=torch.device('cpu'))
    assert 'blocks.0.att.time_state' in w.keys()
    layers = len(w)
    n_embd = w['blocks.0.att.time_state'].shape[0] * w['blocks.0.att.time_state'].shape[1]
    d = [[] for i in range(layers)]
    for i in range(layers):
        d[i].append(torch.zeros(n_embd, dtype=torch.float32))
        d[i].append(w[f'blocks.{i}.att.time_state'].float().transpose(1, 2))
        d[i].append(torch.zeros(n_embd, dtype=torch.float32))

    def pack(x):
        if isinstance(x, torch.Tensor):
            return {'dtype': x.dtype, 'data': x.numpy().tobytes(), 'shape': x.shape}
        elif isinstance(x, torch.dtype):
            return str(x)
        return x

    msgpack.pack(d, open(output_path, 'wb'), default=pack)

if __name__ == '__main__':
    convert_to_fr_state(sys.argv[1], sys.argv[2])
