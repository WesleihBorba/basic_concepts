# Goal:
import threading
import time
import asyncio
import multiprocessing

def greeting_with_sleep(string):
  s = time.perf_counter()
  print(string)
  time.sleep(2)
  print("says hello!")
  elapsed = time.perf_counter() - s
  print("Sequential Programming Elapsed Time: " + str(elapsed) + " seconds")

greeting_with_sleep('Codecademy')

t = threading.Thread(target=greeting_with_sleep, args=('Codecademy',))
t.start()




def greeting_with_sleep(string):
  print(string)
  time.sleep(2)
  print(string + " says hello!")


def main_multiprocessing():
  s = time.perf_counter()
  processes = []
  greetings = ['Codecademy', 'Chelsea', 'Hisham', 'Ashley']
  # add your code here
  for i in range(len(greetings)):
    # create process
    p = multiprocessing.Process(target=greeting_with_sleep, args=(greetings[i],))
    # add process to processes list
    processes.append(p)
    # start process
    p.start()
# join each process
  for p in processes:
    p.join()
  elapsed = time.perf_counter() - s
  print("Multiprocessing Elapsed Time: " + str(elapsed) + " seconds")

main_multiprocessing()



# Concurrency

# Parallelism

# asynchronous
async def hello_async():
  print("hello")
  await asyncio.sleep(3)
  print("how are you?")

asyncio.run(hello_async)

# PAREI AQUI: The Asyncio Module