

class Indexer:
    def __init__(self):
        self.data2index = {}
        self.next_index = 0

    def get_index(self, datum = None):
        if datum not in self.data2index:
            self.data2index[datum] = self.next_index
            self.next_index += 1
        elif datum is None:
            self.data2index[datum] = self.next_index
            self.next_index += 1
        return self.data2index[datum]