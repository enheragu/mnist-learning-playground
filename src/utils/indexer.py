

class Indexer:
    def __init__(self):
        self.data2index = {}
        self.next_index = 0

    def get_index(self, dato = None):
        if dato not in self.data2index:
            self.data2index[dato] = self.next_index
            self.next_index += 1
        elif dato is None:
            self.data2index[dato] = self.next_index
            self.next_index += 1
        return self.data2index[dato]