import time




class Timer:
   def __init__(self):
       self._t0 = time.perf_counter()
       self._marks = {}


   def mark(self, name: str):
       self._marks[name] = time.perf_counter()


   def elapsed_since(self, name: str) -> float:
       if name not in self._marks:
           return 0.0
       return time.perf_counter() - self._marks[name]


   def now(self) -> float:
       return time.perf_counter() - self._t0


   def summary(self) -> dict:
       return {
           "runtime_s": round(self.now(), 3),
           "marks": {
               k: round(v - self._t0, 3)
               for k, v in self._marks.items()
           }
       }

