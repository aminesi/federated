import os
import shutil

for path, subdirs, files in os.walk('results/images'):
    if 'latex' in path:
        continue
    for name in files:
        if name.endswith('.png'):
            file = os.path.join(path, name)
            out = file.replace('results/images/', '').split('/')
            out[0], out[1] = out[1], out[0]
            out = '-'.join(out).replace('adni-defences-0', 'adni-defences')
            out_dir = 'results/images/latex'
            os.makedirs(out_dir, exist_ok=True)
            shutil.copy(file, os.path.join(out_dir, out))
