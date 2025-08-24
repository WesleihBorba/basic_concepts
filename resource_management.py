# Two ways to read Big Data.
# Preventing resource leaks, Preventing crashes, Decreasing the vulnerability of our data on Computer
from contextlib import contextmanager


class WorkWithFile:
    def __init__(self, file_name, mode):
        self.file = file_name
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


with WorkWithFile("big_data_1.txt", "r") as file:
    print(file.read())


# Doing with a library and exception. We can read more than two files
@contextmanager
def open_file_context_lib(file_name, mode):
    open_file = open(file_name, mode)

    try:
        yield open_file

    # Exception Handling
    except Exception as exception:
        print('We hit an error: ' + str(exception))

    finally:
        open_file.close()


with open_file_context_lib('big_data_2.txt', 'w') as opened_file:
    opened_file.sign('We just made a context manager using contex_lib')
