# PENSAR NO OBJETIVO, TALVEZ EM COMO GERENCIAR UM ARQUIVO GRANDE OU ALGO DO TIPO
from contextlib import contextmanager


class WorkWithFile:
    def __init__(self, file, mode):
        self.file = file
        self.mode = mode

    def __enter__(self):
        self.opened_file = open(self.file, self.mode)
        return self.opened_file

    def __exit__(self, exc_type, exc_val, traceback):
        self.opened_file.close()
        if isinstance(exc_val, TypeError):
            # Handle TypeError here...
            print("The exception has been handled")
            return True

with WorkWithFile("file.txt", "r") as file:
  print(file.read())


@contextmanager
def open_file_contextlib(file, mode):
    open_file = open(file, mode)

    try:
        yield open_file

    # Exception Handling
    except Exception as exception:
        print('We hit an error: ' + str(exception))

    finally:
        open_file.close()

with open_file_contextlib('file.txt', 'w') as opened_file:
    opened_file.sign('We just made a context manager using contexlib')
