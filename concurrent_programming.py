# Goal:
import threading
import time
import asyncio
import multiprocessing
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging
import sys

# Logger setting
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Console will show everything

# Handler to console
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class TaskML:
    def __init__(self):
        digits = load_digits()
        X, y = digits.data, digits.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

    def train_and_evaluate(self, name):
        logger.info('Doing a training')
        model = LogisticRegression(max_iter=2000)

        model.fit(self.X_train, self.y_train)
        predict = model.predict(self.X_test)

        acc = accuracy_score(self.y_test, predict)
        logger.debug(f"{name} Accuracy ={acc:.3f}")

    def run_threading(self):
        logger.info('Running threading')
        threads = []  # Line of execution into a process

        logger.info('Creating two threats')
        for i in range(2):
            t = threading.Thread(target=self.train_and_evaluate, args=(f"Thread-{i}",))
            threads.append(t)
            t.start()
        for t in threads:  # Wait threat finish
            t.join()

    def run_multiprocessing(self):
        logger.info('Running Multiprocessing')
        processes = []

        for i in range(2):
            p = multiprocessing.Process(target=self.train_and_evaluate, args=(f"Process-{i}",))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()

    async def async_task(self, name):
        logger.info('Running Asyncio, looping events')
        await asyncio.to_thread(self.train_and_evaluate, name)

    def run_asyncio(self):
        async def main():  # Will run our task
            await asyncio.gather(self.async_task("Async-0"),
                                 self.async_task("Async-1"))
        asyncio.run(main())


if __name__ == "__main__":
    exp = TaskML()

    start = time.time()
    exp.run_threading()
    logger.info(f"Threading time running: {time.time() - start}")

    start = time.time()
    exp.run_multiprocessing()
    logger.info(f"Multiprocessing time running: {time.time() - start}")

    start = time.time()
    exp.run_asyncio()
    logger.info(f"Asyncio time running: {time.time() - start}")
