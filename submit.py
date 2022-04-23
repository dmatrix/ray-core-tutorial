import ray
import time


@ray.remote(num_cpus=1)
class Actor:
    def __init__(self):
        self.a = 1

    def f(self):
#         time.sleep(1)
        self.a += 1
        return self.a


if __name__ == "__main__":
    # connect to the local host with RAY_ADDRESS=127.0.0.1
    ray.init(address="auto")
    actors = [Actor.remote() for _ in range(5)]
    refs = [a.f.remote() for a in actors]
    results = ray.get(refs)
    print(results)


