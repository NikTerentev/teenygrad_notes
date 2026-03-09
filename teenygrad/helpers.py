from typing import Union, Tuple, Iterator, Optional, Final, Any
import os, functools, platform
import numpy as np
from math import prod # noqa: F401 # pylint:disable=unused-import
from dataclasses import dataclass

# Переменная, хранит проверку того, название текущией системы Мак или нет.
# Используется в тестах
OSX = platform.system() == "Darwin"
# Дедупликация - удаляет повторы в списке с сохранением порядка
# (через set порядок не сохранится)
def dedup(x): return list(dict.fromkeys(x))   # retains list orderi
# argfix - приводит к одному виду (если мы передадим просто tuple/list или вложенный list,
# он всё равно приведёт к tuple, тем самым мы можем не задумываться об формате данных в shape).
# >>> argfix(*[1,2,3])
# (1, 2, 3)
# >>> argfix(*[[1,2,3]])
# (1, 2, 3)
def argfix(*x): return tuple(x[0]) if x and x[0].__class__ in (tuple, list) else x
# Если х - число, то возвращаем tuple где x встречается cnt раз, инчае просто x возвращаем
# Проще говоря преобразует число в кортеж нужной длины.
# Например, при передаче размера ядра можно указывать не (2, 2), а просто 2, и тогда
# при использовании этого метода и получится (2, 2).
def make_pair(x:Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]: return (x,)*cnt if isinstance(x, int) else x
# Превращает итераторы каких-то коллекций в плоский список.
def flatten(l:Iterator): return [item for sublist in l for item in sublist]
# Возвращает массив индексов отсортированных элементов из x по возрастанию (сам x не меняется при этом).
# >>> x = [40, 10, 30, 20]
# >>> helpers.argsort(x)
# [1, 3, 2, 0]
def argsort(x): return type(x)(sorted(range(len(x)), key=x.__getitem__)) # https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
# Проверяет, все ли int в tuple.
def all_int(t: Tuple[Any, ...]) -> bool: return all(isinstance(s, int) for s in t)
# ОКРУГЛЯЕТ ДО БЛИЖАЙШЕГО КРАТНОГО amt ВВЕРХ!!
# Скорее всего нужно для выравнивания данных, чтобы размер был кратен байтам amt.
# >>> helpers.round_up(9, 5)
# 10
# >>> helpers.round_up(12, 5)
# 15
def round_up(num, amt:int): return (num+amt-1)//amt * amt

# Достаёт переменные среды, кэшируем чтобы не читать env каждый раз
@functools.lru_cache(maxsize=None)
def getenv(key, default=0): return type(default)(os.getenv(key, default))

DEBUG = getenv("DEBUG")
CI = os.getenv("CI", "") != ""

# Датакласс используемый для представления данных
# frozen=True - иммутабельный после инициализации, можно юзать ключом словаря
# order=True - автоматически генерятся магические методы сравнения
# < > = и тд, позволяющие по полям сравнивать (сначала будут сравниваться по priority
# field так как он первый, что соответственно нам и нужно для апкастинга).
@dataclass(frozen=True, order=True)
class DType:
  # Приоритет для апкастинга (автоматическое повышение типа при операциях)
  # Нужно, чтобы избежать потери точности при смешанных операциях.
  # Например, int + float = float
  priority: int  # this determines when things get upcasted
  # Размер в байтах
  itemsize: int
  # Читаемое название
  name: str
  # Соответствующий тип NumPy
  np: Optional[type]  # TODO: someday this will be removed with the "remove numpy" project
  # Размер для специализированных типов.
  sz: int = 1
  # Репрезентация класса.
  def __repr__(self): return f"dtypes.{self.name}"


# Контейнер для всех типов данных.
class dtypes:
  # staticmethod вверху все чтобы bool не ссылался на dtypes.bool, определённый ниже!
  # Проверяет принадлежность x к целочисленным типам данных.
  @staticmethod # static methds on top, or bool in the type info will refer to dtypes.bool
  def is_int(x: DType)-> bool: return x in (dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64)
  # Проверяет принадлежность x к float типам данных.
  @staticmethod
  def is_float(x: DType) -> bool: return x in (dtypes.float16, dtypes.float32, dtypes.float64)
  # Проверяет принадлежность x к unsigned типам данных.
  @staticmethod
  def is_unsigned(x: DType) -> bool: return x in (dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64)
  # Из переданного типа numpy возвращает DTYPE из словаря согласно имени типа
  # x -> numpy dtype
  @staticmethod
  def from_np(x) -> DType: return DTYPES_DICT[np.dtype(x).name]
  # Инициализация всех основных dtypes
  bool: Final[DType] = DType(0, 1, "bool", np.bool_)
  float16: Final[DType] = DType(9, 2, "half", np.float16)
  # "Алиас" на float16
  half = float16
  float32: Final[DType] = DType(10, 4, "float", np.float32)
  # "Алиас" на float32
  float = float32
  float64: Final[DType] = DType(11, 8, "double", np.float64)
  # "Алиас" на float64
  double = float64
  int8: Final[DType] = DType(1, 1, "char", np.int8)
  int16: Final[DType] = DType(3, 2, "short", np.int16)
  int32: Final[DType] = DType(5, 4, "int", np.int32)
  int64: Final[DType] = DType(7, 8, "long", np.int64)
  uint8: Final[DType] = DType(2, 1, "unsigned char", np.uint8)
  uint16: Final[DType] = DType(4, 2, "unsigned short", np.uint16)
  uint32: Final[DType] = DType(6, 4, "unsigned int", np.uint32)
  uint64: Final[DType] = DType(8, 8, "unsigned long", np.uint64)

  # NOTE: bfloat16 isn't supported in numpy
  bfloat16: Final[DType] = DType(9, 2, "__bf16", None)

# Словарь DTYPE'ов вида: "bool": DType(0, 1, "bool", np.bool_) и т.д...
DTYPES_DICT = {k: v for k, v in dtypes.__dict__.items() if not k.startswith('__') and not callable(v) and not v.__class__ == staticmethod}

# Ненужные типы
PtrDType, ImageDType, IMAGE = None, None, 0  # junk to remove
