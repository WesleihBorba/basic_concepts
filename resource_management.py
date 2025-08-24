# PENSAR NO OBJETIVO, TALVEZ EM COMO GERENCIAR UM ARQUIVO GRANDE OU ALGO DO TIPO

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

# Parei aqui: Handling Exceptions II