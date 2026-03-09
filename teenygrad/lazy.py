from __future__ import annotations
from teenygrad.helpers import DType, dtypes, DEBUG
from teenygrad.ops import UnaryOps, BinaryOps, ReduceOps, TernaryOps, LoadOps
import numpy as np

# ЦПУ буффер (по-видимому, просто заглушка, в tinygrad должна быть реализация).
class RawCPUBuffer:
  def __init__(self, x): self.x = x
  def toCPU(self): return self.x

# Некий буфер для lazy вычислений.
# По сути в teenygrad вычисления на месте происходят (для упрощений).
# В tinygrad скорее всего устроен сложнее и лучше с полноценными ленивыми вычислениями.
# Сам по себе класс представляет ленивую операцию (граф вычислений).
class LazyBuffer:
  device = "CPU"

  # Устанавливает buf как np (numpy) поле инстанса.
  def __init__(self, buf: np.ndarray): self._np = buf

  # В данном проекте/контексте просто возвращает self.
  @property
  def base(self): return self
  # Достаёт dtype из numpy ndarray.
  @property
  def dtype(self): return dtypes.from_np(self._np.dtype)
  # Возвращает реальные данные (в данном случае numpy массив).
  # RawCPUBuffer тут просто заглушка.
  # По сути строим граф вычислений -> "реализуем" -> получаем RawCPUBuffer с данными.
  @property
  def realized(self): return RawCPUBuffer(self._np)
  # Возвращает размеры numpy ndarray.
  @property
  def shape(self): return self._np.shape
  # Представление
  def __repr__(self): return f"<LB {self.shape} {self.dtype}>"

  def schedule(self, seen=None): return []
  def is_unrealized_contiguous_const(self): return False
  def copy_to_device(self, device:str) -> LazyBuffer: return self

  @staticmethod
  def fromCPU(x): return LazyBuffer(x)

  @staticmethod
  def loadop(op, shape, dtype, device, arg=None, src=None) -> LazyBuffer:
    # Если LoadOps.RAND, то дергаем из numpy метод для генерации array рандомных float чисел.
    # Находятся в диапазоне [0.0, 1.0)
    # >>> shape = 3 * 1 * 3 * 3
    # >>> np.random.default_rng(42).random(size=shape)
    # array([0.77395605, 0.43887844, 0.85859792, 0.69736803, 0.09417735,
    #        0.97562235, 0.7611397 , 0.78606431, 0.12811363, 0.45038594,
    #        0.37079802, 0.92676499, 0.64386512, 0.82276161, 0.4434142 ,
    #        0.22723872, 0.55458479, 0.06381726, 0.82763117, 0.6316644 ,
    #        0.75808774, 0.35452597, 0.97069802, 0.89312112, 0.7783835 ,
    #        0.19463871, 0.466721  ])
    if op == LoadOps.RAND: return LazyBuffer(np.random.default_rng(arg).random(size=shape, dtype=dtype.np))
    # Если LoadOps.CONST, то дергаем из numpy метод, который вернет массив размера shape заполненный константой
    # >>> shape = 3 * 1 * 3 * 3
    # >>> np.full(shape=shape, fill_value=3)
    # array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    #        3, 3, 3, 3, 3])
    elif op == LoadOps.CONST: return LazyBuffer(np.full(shape, arg, dtype=dtype.np))
    # Если LoadOps.EMPTY, то numpy создаёт массив указанного shape без инициализации - значения мусорные
    # (то, что было в памяти раньше).
    # >>> shape = 3 * 1 * 3 * 3
    # >>> np.empty(shape=shape)
    # array([1.5e-323, 1.5e-323, 1.5e-323, 1.5e-323, 1.5e-323, 1.5e-323,
    #        1.5e-323, 1.5e-323, 1.5e-323, 1.5e-323, 1.5e-323, 1.5e-323,
    #        1.5e-323, 1.5e-323, 1.5e-323, 1.5e-323, 1.5e-323, 1.5e-323,
    #        1.5e-323, 1.5e-323, 1.5e-323, 1.5e-323, 1.5e-323, 1.5e-323,
    #        1.5e-323, 1.5e-323, 1.5e-323])
    elif op == LoadOps.EMPTY: return LazyBuffer(np.empty(shape, dtype=dtype.np))
    # Иначе - исключение
    else: raise NotImplementedError(op)

  def contiguous(x): return x
  def const(self, x) -> LazyBuffer: return LazyBuffer(np.full_like(self._np, x))

  def cast(self, dtype:DType, bitcast:bool=False): return LazyBuffer(self._np.view(dtype.np) if bitcast else self._np.astype(dtype.np))

  def e(self, op, *srcs:LazyBuffer):
    if DEBUG >= 1: print(op, self, srcs)
    if op == UnaryOps.NEG: ret = -self._np
    elif op == UnaryOps.EXP2: ret = np.exp2(self._np)
    elif op == UnaryOps.LOG2: ret = np.log2(self._np)
    elif op == UnaryOps.SIN: ret = np.sin(self._np)
    elif op == UnaryOps.SQRT: ret = np.sqrt(self._np)
    elif op == BinaryOps.ADD: ret = self._np + srcs[0]._np
    elif op == BinaryOps.SUB: ret = self._np - srcs[0]._np
    elif op == BinaryOps.MUL: ret = self._np * srcs[0]._np
    elif op == BinaryOps.DIV: ret = self._np / srcs[0]._np
    elif op == BinaryOps.MAX: ret = np.maximum(self._np, srcs[0]._np)
    elif op == BinaryOps.CMPLT: ret = self._np < srcs[0]._np
    elif op == TernaryOps.WHERE: ret = np.where(self._np, srcs[0]._np, srcs[1]._np)
    else: raise NotImplementedError(op)
    return LazyBuffer(ret.astype(self.dtype.np if len(srcs) == 0 else max(self.dtype, *[x.dtype for x in srcs]).np, copy=False))

  def r(self, op, new_shape):
    if DEBUG >= 1: print(op, self, new_shape)
    assert len(self.shape) == len(new_shape), "reduce shapes must have same dimensions"
    axis = tuple(i for i,(a,b) in enumerate(zip(self.shape, new_shape)) if a != b)
    if op == ReduceOps.SUM: return LazyBuffer(self._np.sum(axis, dtype=self._np.dtype, keepdims=True))
    elif op == ReduceOps.MAX: return LazyBuffer(self._np.max(axis, keepdims=True))
    else: raise NotImplementedError(op)

  # MovementOps
  def reshape(self, arg): return LazyBuffer(self._np.reshape(arg))
  def expand(self, arg): return LazyBuffer(np.broadcast_to(self._np, arg))
  def shrink(self, arg): return LazyBuffer(self._np[tuple(slice(p[0], p[1], None) for p in arg)])
  def permute(self, arg): return LazyBuffer(self._np.transpose(arg))
  def pad(self, arg): return LazyBuffer(np.pad(self._np, arg))
  def stride(self, arg): return LazyBuffer(self._np[tuple(slice(None, None, i) for i in arg)])
