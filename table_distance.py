def distance(table_1, table_2):
    '''
    modelling the tables as vectors, whose indices are essentially some
    hashing function applied to each (row key, col key) pair, return the
    euclidean distance between them, where euclidean distance is defined as

    sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + ... + (a[n] - b[n])**2)

    assumes that table_1 and table_2 are identical in structure
    '''
    row_keys = table_1.keys()
    cols = list(table_1.values())
    col_keys = cols[0].keys()

    result = 0
    for (row_key, col_key) in zip(row_keys, col_keys):
        delta = (table_1[row_key][col_key] -
                 table_2[row_key][col_key]) ** 2
        result += delta

    return result ** 0.5
