import collections
import itertools
import json
from enum import Enum
from pglast import *
from pglast import ast
from pglast.stream import IndentedStream


class Type(Enum):
    INT = 1
    DECIMAL = 2


class Var:
    def __init__(self, name: str, type: str):
        self.name = name
        self.type = Type[type]

    def __str__(self):
        return f"{self.name}: {self.type}"

    def __repr__(self):
        return self.__str__()


indent = 4
tab = " " * indent


def split_assign(assign: str):
    """
    Splits an assignment statement into the variable name and the expression.
    :param assign: Statement, as a string
    :return: Tuple (var_name, expr)
    """
    # we split on the first equals sign, and then strip the whitespace
    # from the variable name and expression
    assign_exprs = [part.strip() for part in assign.split(":=")]
    assert len(assign_exprs) == 2
    return assign_exprs[0], assign_exprs[1]


class UdfRewriter:
    def __init__(self, tree: json):
        self.tree = tree
        self.vars = []
        self.out = []
        self.populate_vars()
        self.rewrite()
        # print out the flattened program
        print("\n".join(self.flatten_program(self.out)))

    def populate_vars(self):
        for node in self.tree["datums"]:
            if "PLpgSQL_var" not in node:
                continue
            var = node["PLpgSQL_var"]
            var_type = var["datatype"]["PLpgSQL_type"]["typname"]
            if var_type != "UNKNOWN":
                self.vars.append(Var(var["refname"], var_type))

    def flatten_program(self, prog):
        if isinstance(prog, list):
            return [
                subblock if isinstance(subblock, list) else tab + subblock
                for block in prog
                for subblock in self.flatten_program(block)
            ]
        else:
            return [prog]

    def put_declare(self):
        self.out.append("DECLARE")
        block = []
        for var in self.vars:
            block.append(f"{var.name} {var.type.name}[];")
        self.out.append(block)

    def put_batched_sql(self, stmt, block):
        block += self.batch_query(stmt).split("\n")

    def put_action(self):
        self.out.append("BEGIN")
        block = [
            "DROP TABLE IF EXISTS temp;",
            "CREATE TEMP TABLE temp",
            "(",
            ") ON COMMIT DROP;",
            "INSERT INTO temp SELECT * FROM UNNEST(params);",
        ]

        self.put_block(
            self.tree["action"]["PLpgSQL_stmt_block"]["body"],
            self.out,
            header=block,
            top_level=True,
        )

    def put_looped_stmt(self, stmt, block):
        block.append("for i in array_lower(params, 1)..array_upper(params, 1) loop")
        loop_body = []
        self.put_stmt(stmt, loop_body)
        block.append(loop_body)
        block.append("end loop;")

    def put_block(self, tree: list, super_block, header=None, top_level=False):
        """
        Takes in a list of statements, creates a block containing these statements,
        and appends the block to the super_block.
        :param top_level:
        :param tree:
        :param super_block:
        :param header:
        :return:
        """
        if header is not None:
            block = header
        else:
            block = []
        for stmt in tree:
            if top_level:
                self.put_stmt_toplevel(stmt, block)
            else:
                self.put_stmt(stmt, block)
        super_block.append(block)

    def put_stmt(self, stmt, block):
        if "PLpgSQL_stmt_assign" in stmt:
            assign_stmt = stmt["PLpgSQL_stmt_assign"]["expr"]["PLpgSQL_expr"]["query"]
            lhs, rhs = split_assign(assign_stmt)
            block.append(lhs + "[i] := " + rhs + ";")
        elif "PLpgSQL_stmt_execsql" in stmt:
            block.append(stmt["PLpgSQL_stmt_execsql"])
        elif "PLpgSQL_stmt_if" in stmt:
            block.append(
                "if "
                + stmt["PLpgSQL_stmt_if"]["cond"]["PLpgSQL_expr"]["query"]
                + " then"
            )
            if "then_body" in stmt["PLpgSQL_stmt_if"]:
                self.put_block(stmt["PLpgSQL_stmt_if"]["then_body"], block)
            if "else_body" in stmt["PLpgSQL_stmt_if"]:
                block.append("else")
                self.put_block(stmt["PLpgSQL_stmt_if"]["else_body"], block)
            block.append("end if;")
        elif "PLpgSQL_stmt_return" in stmt:
            block.append(
                "ret_vals[i] := "
                + stmt["PLpgSQL_stmt_return"]["expr"]["PLpgSQL_expr"]["query"]
                + ";"
            )
        else:
            raise Exception("Unknown statement type: " + str(stmt))

    def put_stmt_toplevel(self, stmt, block):
        if "PLpgSQL_stmt_execsql" in stmt:
            self.put_batched_sql(stmt["PLpgSQL_stmt_execsql"], block)
        else:
            self.put_looped_stmt(stmt, block)

    def generate_into_clause(self, stmt):
        assert len(stmt["target"]["PLpgSQL_row"]["fields"]) == 1

        var_name = stmt["target"]["PLpgSQL_row"]["fields"][0]["name"]
        return ast.IntoClause(
            rel=ast.RangeVar(relname=var_name, inh=True),
            onCommit=enums.OnCommitAction.ONCOMMIT_NOOP,
        )

    def batch_query(self, stmt):
        """
        Batches a UDF query. For example, given the query "SELECT x FROM table" and the key "k",
        this function will return "SELECT ARRAY_AGG(x order by k) FROM (SELECT * FROM params, LATERAL (SELECT a FROM table) dt0) dt1".
        :param query:
        :return:
        """
        query = stmt["sqlstmt"]["PLpgSQL_expr"]["query"]
        select_stmt = parse_sql(query)[0].stmt
        assert isinstance(select_stmt, ast.SelectStmt)

        target_aggs = []
        num_targets = len(select_stmt.targetList)
        for i, target in enumerate(select_stmt.targetList):
            target_aggs.append(f"agg_{i}")
            target.name = target_aggs[i]

        subselect = ast.RangeSubselect(
            lateral=True, subquery=select_stmt, alias=ast.Alias("dt1")
        )
        temp_table = ast.RangeVar(relname="temp", inh=True)
        from_star = ast.ColumnRef((ast.A_Star(),))
        from_target = ast.ResTarget(val=from_star)
        outer_target_list = (from_target,)

        array_aggs = []
        for i in range(num_targets):
            func_call = ast.FuncCall(
                funcname=(ast.String("array_agg"),),
                args=(ast.RangeVar(relname=f"agg_{i}", inh=True),),
            )
            array_agg = ast.ResTarget(val=func_call)
            array_aggs.append(array_agg)

        new_query = ast.SelectStmt(
            targetList=tuple(array_aggs), fromClause=(temp_table, subselect)
        )

        new_query.intoClause = self.generate_into_clause(stmt)

        return IndentedStream()(new_query) + ";"

    def put_return(self):
        self.out.append("RETURN QUERY SELECT * FROM UNNEST(ret_vals);")

    def rewrite(self):
        self.put_declare()
        self.put_action()
        self.put_return()
