import copy
from enum import Enum
from pglast import *
from pglast import ast
from pglast.enums import OnCommitAction
from pglast.stream import IndentedStream
from pglast.visitors import Visitor
from typing import List


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


def gen_assign_stmt(lhs: str, rhs: str):
    return lhs + " := " + rhs + ""


class VarToArrayRefRewriter(Visitor):
    """
    When there is a query that can't be batched, rewrite the
    query to reference var[i] instead of var for each local
    variable that it references.
    """

    def __init__(self, local_vars: set):
        self.local_vars = local_vars

    def visit_ColumnRef(self, parent, node: ast.ColumnRef):
        if node.fields[0].val in self.local_vars:
            return ast.A_Indirection(
                arg=node,
                indirection=[
                    ast.A_Indices(uidx=ast.ColumnRef(fields=[ast.String("i")]))
                ],
            )


class FunctionBodyRewriter(Visitor):
    """
    Replaces the function body in the SQL AST, where it is represented as a string,
    with the new rewritten body.
    """

    def __init__(self, new_body: str):
        self.new_body = new_body

    def visit_DefElem(self, parent, node: ast.DefElem):
        if node.defname == "as":
            return ast.DefElem(defname="as", arg=(ast.String(self.new_body),))


class UdfRewriter:
    def __init__(self, f: str):
        self.sql_tree = parse_sql(f)[0].stmt
        self.tree = parse_plpgsql(f)[0]["PLpgSQL_function"]
        self.vars = {}  # maps varnos to variable names
        self.out = []  # (nested) list of statements to output
        self.params = [Param(param) for param in self.sql_tree.parameters]

        self.rewrite_header()
        self.populate_vars()
        self.rewrite_body()
        self.replace_function_body("\n".join(self.flatten_program(self.out)))
        self.output_sql = IndentedStream()(self.sql_tree) + ";"

    def output(self) -> str:
        return self.output_sql

    def replace_function_body(self, new_body):
        """
        Replaces the function body in the SQL AST, where it is represented as a string,
        with the new rewritten body.
        :param new_body:
        :return:
        """
        FunctionBodyRewriter(new_body)(self.sql_tree)

    def populate_vars(self):
        """
        Parses the variables declared in the function and stores them in self.vars.
        :return:
        """
        for varno, node in enumerate(self.tree["datums"]):
            if "PLpgSQL_var" not in node:
                continue
            var = node["PLpgSQL_var"]
            var_type = var["datatype"]["PLpgSQL_type"]["typname"]
            if var_type != "UNKNOWN":
                self.vars[varno] = Var(var["refname"], var_type)

    def flatten_program(self, prog) -> List[str]:
        """
        Flattens a program, which is a (nested) list of strings, into a single list of strings,
        indenting statements as necessary.
        :param prog: Program, as a string or (nested) list of strings
        :return: Flattened program, as a list of strings
        """
        if isinstance(prog, list):
            return [
                subblock if isinstance(subblock, list) else tab + subblock
                for block in prog
                for subblock in self.flatten_program(block)
            ]
        else:
            return [prog]

    def put_declare(self):
        """
        Puts the DECLARE statement at the beginning of the function (declares local variables).
        """
        self.out.append("")
        self.out.append("DECLARE")
        block = []
        for var in self.vars.values():
            block.append(f"{var.name} {var.type}[];")
        block.append(f"ret_vals {self.sql_tree.returnType.names[1].val}[];")
        block.append("returned BOOL[];")
        self.out.append(block)

    def put_batched_sql(self, stmt: dict, block: list):
        """
        Takes in a SQL statement, transforms it into a batched query, and appends it to the given output block.
        :param stmt: PL/pgSQL AST for a SQL statement inside a UDF
        :param block: List of statements to append output to
        """
        # Make sure we are only SELECTing into one variable
        assert len(stmt["target"]["PLpgSQL_row"]["fields"]) == 1
        query_str = stmt["sqlstmt"]["PLpgSQL_expr"]["query"]
        into_var = stmt["target"]["PLpgSQL_row"]["fields"][0]["name"]
        batched_query = self.batch_query(query_str, into=into_var).split("\n")
        batched_query[-1] += ";"
        block += batched_query

    def put_action(self):
        self.out.append("BEGIN")

        block = []

        temp_table_cols = []
        for param in self.params:
            column = ast.ColumnDef(colname=param.name, typeName=param.type)
            temp_table_cols.append(column)
        create_table_stmt = ast.CreateStmt(
            relation=ast.RangeVar(
                relname="temp",
                inh=True,
                relpersistence="t",
            ),
            tableElts=temp_table_cols,
            oncommit=OnCommitAction.ONCOMMIT_DROP,
        )
        block += (IndentedStream()(create_table_stmt) + ";").split("\n")

        # INSERT INTO temp SELECT unnest(param1_batch), unnest(param2_batch), ...
        insert_stmt = ast.InsertStmt(
            relation=ast.RangeVar(relname="temp", inh=True, relpersistence="p"),
            selectStmt=ast.SelectStmt(
                targetList=[
                    ast.ResTarget(
                        val=ast.FuncCall(
                            funcname=[ast.String("unnest")],
                            args=[
                                ast.ColumnRef(
                                    fields=[ast.String(param.name + "_batch")]
                                )
                                for param in self.params
                            ],
                        )
                    )
                ]
            ),
        )
        block += (IndentedStream()(insert_stmt) + ";").split("\n")

        self.put_block(
            self.tree["action"]["PLpgSQL_stmt_block"]["body"],
            self.out,
            header=block,
            footer="RETURN ret_vals;",
            top_level=True,
        )

    def put_looped_stmt(self, stmt, block):
        block.append(
            f"FOR i IN ARRAY_LOWER({self.params[0].name}_batch, 1)..ARRAY_UPPER({self.params[0].name}_batch, 1) LOOP"
        )
        loop_body = []
        self.put_stmt(stmt, loop_body)
        block.append(loop_body)
        block.append("END LOOP;")

    def put_block(
        self, tree: list, super_block, header=None, footer=None, top_level=False
    ):
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
        if footer is not None:
            block.append(footer)
        super_block.append(block)

    def put_stmt(self, stmt, block):
        if "PLpgSQL_stmt_assign" in stmt:
            sql = stmt["PLpgSQL_stmt_assign"]["expr"]["PLpgSQL_expr"]["query"]
            lhs = self.vars[stmt["PLpgSQL_stmt_assign"]["varno"]]
            rhs_ast = parse_sql(sql)
            VarToArrayRefRewriter(self.get_local_var_names())(rhs_ast)
            block.append(lhs.name + "[i] := (" + IndentedStream()(rhs_ast) + ");")
        elif "PLpgSQL_stmt_execsql" in stmt:
            block.append(stmt["PLpgSQL_stmt_execsql"])
        elif "PLpgSQL_stmt_if" in stmt:
            sql = stmt["PLpgSQL_stmt_if"]["cond"]["PLpgSQL_expr"]["query"]
            cond_ast = parse_sql(sql)
            VarToArrayRefRewriter(self.get_local_var_names())(cond_ast)
            block.append("IF (" + IndentedStream()(cond_ast) + ") THEN")
            if "then_body" in stmt["PLpgSQL_stmt_if"]:
                self.put_block(stmt["PLpgSQL_stmt_if"]["then_body"], block)
            if "else_body" in stmt["PLpgSQL_stmt_if"]:
                block.append("ELSE")
                self.put_block(stmt["PLpgSQL_stmt_if"]["else_body"], block)
            block.append("END IF;")
        elif "PLpgSQL_stmt_return" in stmt:
            if "expr" not in stmt["PLpgSQL_stmt_return"]:
                # Implicit return, e.g.
                # IF () THEN ... RETURN x; ELSE ... RETURN y; END IF;
                # <implicit return>
                #
                # We don't codegen anything in this case.
                return

            sql = stmt["PLpgSQL_stmt_return"]["expr"]["PLpgSQL_expr"]["query"]
            rhs_ast = parse_sql(sql)
            VarToArrayRefRewriter(self.get_local_var_names())(rhs_ast)
            block.append("IF returned[i] IS NULL THEN")
            block.append(
                [
                    "ret_vals[i] := (" + IndentedStream()(rhs_ast) + ");",
                    "returned[i] := TRUE;",
                ]
            )
            block.append("END IF;")
        else:
            raise Exception("Unknown statement type: " + str(stmt))

    def put_stmt_toplevel(self, stmt, block):
        if "PLpgSQL_stmt_execsql" in stmt:
            self.put_batched_sql(stmt["PLpgSQL_stmt_execsql"], block)
        elif "PLpgSQL_stmt_assign" in stmt:
            assign_stmt = stmt["PLpgSQL_stmt_assign"]["expr"]["PLpgSQL_expr"]["query"]
            batched_sql = self.batch_query(assign_stmt).split("\n")
            varno = stmt["PLpgSQL_stmt_assign"]["varno"]
            batched_sql[0] = "(" + batched_sql[0]
            batched_sql[-1] = batched_sql[-1] + ");"
            batched_sql[0] = gen_assign_stmt(self.vars[varno].name, batched_sql[0])
            block += batched_sql
        else:
            self.put_looped_stmt(stmt, block)

    def generate_into_clause(self, var_name: str):
        """
        Generates an INTO clause equivalent to "INTO var_name".
        :param var_name:
        :return:
        """
        return ast.IntoClause(
            rel=ast.RangeVar(relname=var_name, inh=True),
            onCommit=enums.OnCommitAction.ONCOMMIT_NOOP,
        )

    def batch_query(self, query: str, into=None):
        """
        Batches a UDF query. For example, given the query "SELECT x FROM table" and the key "k",
        this function will return "SELECT ARRAY_AGG(x order by k) FROM (SELECT * FROM params, LATERAL (SELECT a FROM table) dt0) dt1".
        :param query:
        :return:
        """
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

        if into is not None:
            new_query.intoClause = self.generate_into_clause(into)

        return IndentedStream()(new_query)

    def get_local_var_names(self):
        return set([var.name for var in self.vars.values()])

    def rewrite_body(self):
        self.put_declare()
        self.put_action()
        self.out.append("END;")

    def rewrite_header(self):
        """
        Converts the old function header to a batched function header, mainly by
        making scalar parameters into arrays.
        """
        for param in self.sql_tree.parameters:
            param.argType = copy.copy(param.argType)
            param.argType.arrayBounds = [ast.Integer(-1)]
            param.name = param.name + "_batch"

        # Change the return type to an array
        self.sql_tree.returnType.arrayBounds = [ast.Integer(-1)]

        # We don't want to accidentally replace an existing function
        self.sql_tree.replace = False

        # Change the function name
        self.original_func_name = self.sql_tree.funcname[0].val
        self.sql_tree.funcname = [ast.String(self.sql_tree.funcname[0].val + "_batch")]
