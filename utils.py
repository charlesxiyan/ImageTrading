from __init__ import *

# * context manager decorator is used to allocate memories/resources properly
# ? Please note "yield" here plays a role of partitioning the program, if timer() is called
# ? in a with-statement and there are still commands within the with-statement, yield will 
# ? pause timer() when it reaches there, i.e. so far only running time.time(), and then run
# ? those commands and then come back to run the left codes, i.e. elapsed = ......print(...)

@contextmanager
def timer(name: str, _align):
    s = time.time()
    yield
    elapsed = time.time() - s
    print(
      f"{'[' + name + ']':{_align}} | {time.strftime('%Y-%m-%d %H:%M:%S')} Done | Using {elapsed: .3f} seconds"
    )