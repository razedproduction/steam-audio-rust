"""Runs upstream's sample_binaural (core/src/samples/sample_binaural.cpp) on a golden input.

    python tools/golden/run_sample.py <sample executable> <input .npy> <output .npy>

The sample reads `inputaudio.raw` and writes `outputaudio.raw` (raw 32-bit floats) in its working directory, so this
runs it in a temporary directory: it writes the input there, runs the sample, and saves its output as a 1-D float32
`.npy`. Build the executable from tools/golden/golden.cmake: `sample_binaural_upstream` (upstream's library) or
`sample_binaural_port` (the port's `phonon` library).
"""

import array
import ast
import os
import subprocess
import sys
import tempfile


def read_npy_f32(path):
    with open(path, 'rb') as f:
        data = f.read()
    if data[:6] != b'\x93NUMPY':
        raise ValueError(f'{path}: not a .npy file')
    if data[6] == 1:
        header_len = int.from_bytes(data[8:10], 'little')
        start = 10
    else:
        header_len = int.from_bytes(data[8:12], 'little')
        start = 12
    header = ast.literal_eval(data[start:start + header_len].decode('latin1'))
    if header['descr'] != '<f4' or header['fortran_order'] or len(header['shape']) != 1:
        raise ValueError(f'{path}: expected a 1-D little-endian float32 array, got {header}')
    values = array.array('f')
    values.frombytes(data[start + header_len:])
    return values


def write_npy_f32(path, values):
    header = "{'descr': '<f4', 'fortran_order': False, 'shape': (%d,), }" % len(values)
    # Pad so that the data starts on a multiple of 64 bytes, newline-terminated, as numpy does.
    total = 10 + len(header) + 1
    header += ' ' * ((64 - total % 64) % 64) + '\n'
    with open(path, 'wb') as f:
        f.write(b'\x93NUMPY\x01\x00')
        f.write(len(header).to_bytes(2, 'little'))
        f.write(header.encode('latin1'))
        f.write(values.tobytes())


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    exe, input_path, output_path = (os.path.abspath(p) for p in sys.argv[1:])
    samples = read_npy_f32(input_path)
    with tempfile.TemporaryDirectory() as work:
        with open(os.path.join(work, 'inputaudio.raw'), 'wb') as f:
            f.write(samples.tobytes())
        subprocess.run([exe], cwd=work, check=True)
        output = array.array('f')
        with open(os.path.join(work, 'outputaudio.raw'), 'rb') as f:
            output.frombytes(f.read())
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_npy_f32(output_path, output)
    print(f'{len(samples)} input samples, {len(output)} output samples -> {output_path}')


if __name__ == '__main__':
    main()
