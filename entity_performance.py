def get_entities(x):
    entities = []
    buf = ['O', 0, 0]
    for i, j in enumerate(x):
        if buf[0] == 'O':
            if j != 'O':
                buf = [j[:-2], i, i + 1]
        else:
            if j == 'O':
                entities.append(tuple(buf))
                buf = ['O', 0, 0]
            elif j[-1] == 'S':
                entities.append(tuple(buf))
                entities.append((j[:-2], i, i + 1))
                buf = ['O', 0, 0]
            elif j[-1] == 'B' or buf[0] != j[:-2]:
                entities.append(tuple(buf))
                buf = [j[:-2], i, i + 1]
            else:
                buf[-1] = i + 1
    if buf[0] != 'O':
        entities.append(tuple(buf))
    return entities


def performance(predictions, answers):
    A = 0
    B = 0
    C = 0
    for p, a in zip(predictions, answers):
        entities_p = get_entities(p[: len(a)])
        entities_a = get_entities(a)

        A += len(set(entities_p) & set(entities_a))
        B += len(entities_p)
        C += len(entities_a)

    precision = 1.0 * A / B if B > 0 else 0
    recall = 1.0 * A / C if C > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1


if __name__ == "__main__":
    p = [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'POS-I', 'O', 'O', 'O', 'O', 'O',
          'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'POS-S', 'POS-I', 'O', 'O', 'O', 'O', 'POS-I', 'POS-I', 'POS-B',
          'O', 'O', 'O', 'POS-S', 'POS-S', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
          'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
          'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
          'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
          'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
          'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
          'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
          'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']] * 10
    a = [['O', 'PER-B', 'PER-I', 'PER-E', 'PER-S', 'O', 'O', 'O', 'O', 'O', 'POS-B', 'POS-E', 'O', 'O', 'O', 'O', 'O',
          'O',
          'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'POS-B', 'POS-E', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
          'O', 'O', 'O', 'POS-B', 'POS-E', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
          'O', 'O', 'O', 'O', 'O', 'O', 'O']] * 10

    r = performance(p, a)
    print(r)
