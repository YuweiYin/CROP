from sys import argv
import os


e2x, x2e, x2x = [], [], []
for filename in os.listdir(argv[1]):
    if not filename.startswith('bleu.'):
        continue
    if '-en' in filename:
        for line in open(os.path.join(argv[1], filename)):
            if line.startswith('BLEU+case'):
                x2e.append(float(line.split(' ')[2]))
    elif 'en-' in filename:
        for line in open(os.path.join(argv[1], filename)):
            if line.startswith('BLEU+case'):
                e2x.append(float(line.split(' ')[2]))
    else:
        for line in open(os.path.join(argv[1], filename)):
            if line.startswith('BLEU+case'):
                x2x.append(float(line.split(' ')[2]))

if len(e2x) > 0:
    print('E-X: ', sum(e2x) * 1.0 / len(e2x))

if len(x2e) > 0:
    print('X-E: ', sum(x2e) * 1.0 / len(x2e))
    
if len(x2x) > 0:
    print('X-X: ', sum(x2x) * 1.0 / len(x2x))