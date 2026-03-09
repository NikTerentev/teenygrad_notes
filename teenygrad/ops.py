from enum import Enum, auto
from typing import Optional

# Унарные операции: 1 Тензор -> 1 Тензор.
class UnaryOps(Enum): NOOP = auto(); EXP2 = auto(); LOG2 = auto(); CAST = auto(); SIN = auto(); SQRT = auto(); RECIP = auto(); NEG = auto() # noqa: E702
# Бинарные операции: 2 Тензора -> 1 Тензор.
class BinaryOps(Enum): ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto(); MAX = auto(); MOD = auto(); CMPLT = auto() # noqa: E702
# Редукция (уменьшают размерность).
class ReduceOps(Enum): SUM = auto(); MAX = auto() # noqa: E702
# Тернарные операции (3 Тензора).
class TernaryOps(Enum): MULACC = auto(); WHERE = auto() # noqa: E702
# Перемещение данных (форма/расположение).
class MovementOps(Enum): RESHAPE = auto(); PERMUTE = auto(); EXPAND = auto(); PAD = auto(); SHRINK = auto(); STRIDE = auto() # noqa: E702
# Загрузка/создание данных.
class LoadOps(Enum): EMPTY = auto(); RAND = auto(); CONST = auto(); FROM = auto(); CONTIGUOUS = auto(); CUSTOM = auto() # noqa: E702

# Абстракция вычислительного устройства.
class Device:
  # CPU - девайс по умолчанию.
  DEFAULT = "CPU"
  # Список зарегистрированных устройств.
  _buffers = ["CPU"]
  # Приводит к каноническому виду (в данном случае строка)
  @staticmethod
  def canonicalize(device:Optional[str]) -> str: return "CPU"
