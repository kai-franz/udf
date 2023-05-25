from pglast import parse_plpgsql
from utils import Var
from enum import Enum, auto


class UdfStmtType(Enum):
    ASSIGN = auto()
    EXEC_SQL = auto()
    IF = auto()
    RETURN = auto()


class UdfStmt:
    pass


class AssignStmt(UdfStmt):
    def __init__(self, stmt):
        self.sql = stmt["PLpgSQL_stmt_assign"]["expr"]["PLpgSQL_expr"]["query"]
        self.varno = stmt["PLpgSQL_stmt_assign"]["varno"]


class ExecSqlStmt(UdfStmt):
    def __init__(self, stmt):
        self.sql = stmt["PLpgSQL_stmt_execsql"]


class IfStmt(UdfStmt):
    def __init__(self, stmt):
        self.sql = stmt["PLpgSQL_stmt_if"]["cond"]["PLpgSQL_expr"]["query"]
        if "then_body" in stmt["PLpgSQL_stmt_if"]:
            self.then_body = [
                parse_stmt(s) for s in stmt["PLpgSQL_stmt_if"]["then_body"]
            ]
        else:
            self.then_body = None
        if "else_body" in stmt["PLpgSQL_stmt_if"]:
            self.else_body = [
                parse_stmt(s) for s in stmt["PLpgSQL_stmt_if"]["else_body"]
            ]
        else:
            self.else_body = None


class ReturnStmt(UdfStmt):
    def __init__(self, stmt):
        if "expr" not in stmt["PLpgSQL_stmt_return"]:
            self.sql = None
        else:
            self.sql = stmt["PLpgSQL_stmt_return"]["expr"]["PLpgSQL_expr"]["query"]


def parse_stmt(stmt):
    """
    Parses a statement dict into a UdfStmt object.
    """

    if "PLpgSQL_stmt_assign" in stmt:
        return AssignStmt(stmt)
    elif "PLpgSQL_stmt_execsql" in stmt:
        return ExecSqlStmt(stmt)
    elif "PLpgSQL_stmt_if" in stmt:
        return IfStmt(stmt)
    elif "PLpgSQL_stmt_return" in stmt:
        return ReturnStmt(stmt)

    raise Exception(f"Unknown statement type: {stmt}")


class Udf:
    def __init__(self, f: str):
        self.tree = parse_plpgsql(f)[0]["PLpgSQL_function"]
        self.vars = {}

        for varno, node in enumerate(self.tree["datums"]):
            if "PLpgSQL_var" not in node:
                continue
            var = node["PLpgSQL_var"]
            var_type = var["datatype"]["PLpgSQL_type"]["typname"]
            if var_type != "UNKNOWN":
                self.vars[varno] = Var(var["refname"], var_type)

        self.body = [
            parse_stmt(s) for s in self.tree["action"]["PLpgSQL_stmt_block"]["body"]
        ]
