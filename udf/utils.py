from pglast import ast
from enum import Enum


class BaseType(Enum):
    INT = 1
    DECIMAL = 2
    FLOAT = 3
    VARCHAR = 4
    # handle cases like DECIMAL(10, 2)


class Type:
    def __init__(self, type_str: str):
        self.type_str = type_str
        if "(" in type_str:
            base_type = type_str.split("(")[0]
        else:
            base_type = type_str
        self.base_type = BaseType[base_type]

    def __str__(self):
        return self.type_str


class Var:
    def __init__(self, name: str, type: str):
        self.name = name
        self.type = Type(type)

    def __str__(self):
        return f"{self.name}: {self.type}"

    def __repr__(self):
        return self.__str__()


class Param:
    def __init__(self, param: ast.FunctionParameter):
        self.name = param.name
        self.type = param.argType
        print(param)


indent = 4
tab = " " * indent

TEMP_TABLE_NAME = "temp"
TEMP_KEY_NAME = "temp_key"
