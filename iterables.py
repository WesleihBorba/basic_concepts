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

# Method send(), pensar em como usar
def count_generator():
  while True:
    n = yield
    print(n)

my_generator = count_generator()
next(my_generator) # 1st Iteration Output:
next(my_generator) # 2nd Iteration Output: None
my_generator.send(3) # 3rd Iteration Output: 3
next(my_generator) # 4th Iteration Output: None

# throw(), também pensar em algo
def generator():
  i = 0
  while True:
    yield i
    i += 1

my_generator = generator()
for item in my_generator:
    if item == 3:
        my_generator.throw(ValueError, "Bad value given")


# Method close too
def generator():
  i = 0
  while True:
    yield i
    i += 1

my_generator = generator()
next(my_generator)
next(my_generator)
my_generator.close()
next(my_generator) # raises StopGenerator exception

# Connection generators
def cs_courses():
    yield 'Computer Science'
    yield 'Artificial Intelligence'

def art_courses():
    yield 'Intro to Art'
    yield 'Selecting Mediums'


def all_courses():
    yield from cs_courses()
    yield from art_courses()

combined_generator = all_courses()

# Generator Pipelines

def number_generator():
    i = 0
    while True:
        yield i
        i += 1


def even_number_generator(numbers):
    for n in numbers:
        if n % 2 == 0:
            yield n


even_numbers = even_number_generator(number_generator())

for e in even_numbers:
    print(e)
    if e == 100:
        break
