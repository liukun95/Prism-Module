import re


def f(filename_in, filename_out):
    f_in = open(filename_in)
    f_out = open(filename_out, 'w')
    data = [[], []]
    buffer = []
    for l in f_in:
        if l == '\n':
            if '-docstart-' not in data[0] and '-DOCSTART-' not in data[0]:
                if len(buffer) > 0:
                    if len(buffer) == 1:
                        buffer[0] = buffer[0][:-1] + 'S'
                    else:
                        buffer[0] = buffer[0][:-1] + 'B'
                        buffer[-1] = buffer[-1][:-1] + 'E'
                    data[1].extend(buffer)

                assert len(data[0]) == len(data[1])

                if len(data[0]) > 0:
                    f_out.write(' '.join(data[1]) + '\t' + ' '.join(data[0]) + '\n')
            data = [[], []]
            buffer = []
        elif l != '\n':
            l = l[:-1].split(' ')
            word = l[0]
            tag = l[-1]

            word = re.sub(r'[0-9]+', '0', word)
            if len(re.findall(r'[a-zA-Z]', word)) > 0 and len(re.findall(r'[0-9]', word)) > 0:
                word = re.sub(r'[a-z]', 'a', word)
                word = re.sub(r'[A-Z]', 'A', word)
            data[0].append(word)

            if tag == 'O':
                if len(buffer) > 0:
                    if len(buffer) == 1:
                        buffer[0] = buffer[0][:-1] + 'S'
                    else:
                        buffer[0] = buffer[0][:-1] + 'B'
                        buffer[-1] = buffer[-1][:-1] + 'E'
                    data[1].extend(buffer)
                    buffer = []
                data[1].append(tag)
            elif tag[0] == 'B' or (len(buffer) > 0 and buffer[0][:-2] != tag[2:]):
                if len(buffer) > 0:
                    if len(buffer) == 1:
                        buffer[0] = buffer[0][:-1] + 'S'
                    else:
                        buffer[0] = buffer[0][:-1] + 'B'
                        buffer[-1] = buffer[-1][:-1] + 'E'
                    data[1].extend(buffer)
                buffer = [tag[2:] + '-I']
            else:
                buffer.append(tag[2:] + '-I')

    if len(buffer) > 0:
        if len(buffer) == 1:
            buffer[0] = buffer[0][:-1] + 'S'
        else:
            buffer[0] = buffer[0][:-1] + 'B'
            buffer[-1] = buffer[-1][:-1] + 'E'
        data[1].extend(buffer)
    if len(data[0]) > 0:
        assert len(data[0]) == len(data[1])
        f_out.write(' '.join(data[1]) + '\t' + ' '.join(data[0]) + '\n')


f('data/eng.train', 'data/train.tsv')
f('data/eng.testa', 'data/dev.tsv')
f('data/eng.testb', 'data/test.tsv')
