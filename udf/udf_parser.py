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
    def __init__(self, sql, varno):
        self.sql = sql
        self.varno = varno

    @classmethod
    def from_dict(cls, stmt):
        return cls(
            stmt["PLpgSQL_stmt_assign"]["expr"]["PLpgSQL_expr"]["query"],
            stmt["PLpgSQL_stmt_assign"]["varno"],
        )


class ExecSqlStmt(UdfStmt):
    def __init__(self, sql, into):
        self.sql = sql
        self.into = into

    @classmethod
    def from_dict(cls, stmt):
        sql = stmt["PLpgSQL_stmt_execsql"]["sqlstmt"]["PLpgSQL_expr"]["query"]
        if "target" in stmt["PLpgSQL_stmt_execsql"]["sqlstmt"]:
            into = stmt["PLpgSQL_stmt_execsql"]["sqlstmt"]["target"]["PLpgSQL_row"][
                "fields"
            ][-1]["name"]
        else:
            into = None
        return cls(sql, into)


class IfStmt(UdfStmt):
    def __init__(self, cond, then_body, else_body):
        self.sql = cond
        self.then_body = then_body
        self.else_body = else_body

    @classmethod
    def from_dict(cls, stmt):
        cond = stmt["PLpgSQL_stmt_if"]["cond"]["PLpgSQL_expr"]["query"]
        if "then_body" in stmt["PLpgSQL_stmt_if"]:
            then_body = [parse_stmt(s) for s in stmt["PLpgSQL_stmt_if"]["then_body"]]
        else:
            then_body = None
        if "else_body" in stmt["PLpgSQL_stmt_if"]:
            else_body = [parse_stmt(s) for s in stmt["PLpgSQL_stmt_if"]["else_body"]]
        else:
            else_body = None
        return cls(cond, then_body, else_body)


class ReturnStmt(UdfStmt):
    def __init__(self, sql):
        self.sql = sql

    @classmethod
    def from_dict(cls, stmt):
        if "expr" not in stmt["PLpgSQL_stmt_return"]:
            ret_val = None
        else:
            ret_val = stmt["PLpgSQL_stmt_return"]["expr"]["PLpgSQL_expr"]["query"]
        return cls(ret_val)


def parse_stmt(stmt):
    """
    Parses a statement dict into a UdfStmt object.
    """

    if "PLpgSQL_stmt_assign" in stmt:
        return AssignStmt.from_dict(stmt)
    elif "PLpgSQL_stmt_execsql" in stmt:
        return ExecSqlStmt.from_dict(stmt)
    elif "PLpgSQL_stmt_if" in stmt:
        return IfStmt.from_dict(stmt)
    elif "PLpgSQL_stmt_return" in stmt:
        return ReturnStmt.from_dict(stmt)

    raise Exception(f"Unknown statement type: {stmt}")


def count_if_stmts(stmt: UdfStmt):
    if isinstance(stmt, IfStmt):
        return (
            1
            + sum(count_if_stmts(s) for s in stmt.then_body)
            + sum(count_if_stmts(s) for s in stmt.else_body)
        )
    return 0


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

        self.num_if_stmts = sum(count_if_stmts(s) for s in self.body)
