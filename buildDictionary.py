
def builder(filename):
    with open(filename) as f:
        dictionary = set()
        for line in f.readlines():
            dictionary = dictionary.union(line.strip().split('|'))
    return dictionary