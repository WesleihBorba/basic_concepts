# Iterable in big data
import itertools

sku_list = [7046538, 8289407, 9056375, 2308597]


class IterableBigData:
    def __init__(self, data):
        self.big_data = data

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.big_data):
            self.index += 1
            return self.big_data[self.index - 1]
        else:
            raise StopIteration


iterable_counter = IterableBigData(sku_list)
for i in iterable_counter:
    print(i)

# PAREI AQUI: Generator Methods: send()
# yield
# next()
# Generator comprehension - Need to use loop to access
a_generator = (i*i for i in range(4))