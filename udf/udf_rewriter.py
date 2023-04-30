import copy
from enum import Enum
from pglast import *
from pglast import ast, scan
from pglast.enums import OnCommitAction, ConstrType, A_Expr_Kind, SubLinkType
from pglast.stream import IndentedStream
from pglast.visitors import Visitor
from typing import List

from udf.schema import Schema
from udf.planner_main import Planner
from udf.utils import *


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


class FromClauseFinder(Visitor):
    """
    Finds the FROM clause in a query.
    """

    def __init__(self):
        self.from_clause = False

    def visit_SelectStmt(self, parent, node: ast.SelectStmt):
        if node.fromClause:
            self.from_clause = True


class ColumnRefFinder(Visitor):
    """
    Keeps track of all ColumnRefs in a query.
    """

    def __init__(self):
        self.column_refs = set()

    def visit_ColumnRef(self, parent, node: ast.ColumnRef):
        self.column_refs.add(node.fields[-1].val.lower())


def get_column_refs(query_ast: ast.Node):
    """
    Returns a set of all column references in the query.
    """
    finder = ColumnRefFinder()
    finder(query_ast)
    return finder.column_refs


class ColumnRefReplacer(Visitor):
    """
    Replaces ColumnRefs in a query according to the given replacement dictionary.
    """

    def __init__(self, replacement_dict: dict):
        self.replacement_dict = replacement_dict

    def visit_ColumnRef(self, parent, node: ast.ColumnRef):
        if node.fields[-1].val.lower() in self.replacement_dict:
            return ast.ColumnRef(
                fields=[ast.String(self.replacement_dict[node.fields[-1].val.lower()])]
            )


class UdfRewriter:
    def __init__(self, f: str, schema: Schema, remove_laterals=False):
        self.local_var_refs = set()
        self.vars = {}  # maps varnos to variable names
        self.out = []  # (nested) list of statements to output
        self.schema = schema
        self.remove_laterals = remove_laterals
        self.sql_tree = parse_sql(f)[0].stmt
        self.tree = parse_plpgsql(f)[0]["PLpgSQL_function"]
        self.params = [Param(param) for param in self.sql_tree.parameters]

        self.rewrite_header()
        self.populate_vars()
        self.local_var_names = set(var.name.lower() for var in self.vars.values())
        self.init_local_var_refs()

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
        Flattens a program, which is a (nested) list of strings, into a single list
        of strings, indenting statements as necessary.
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
        Puts the DECLARE statement at the beginning of the function,
        which declares local variables for the main block.
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
        Takes in a SQL statement, transforms it into a batched query, and appends it
        to the given output block.
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
        for var_name in sorted(list(self.local_var_refs)):
            var_type = None
            for var in self.vars.values():
                if var.name == var_name:
                    var_type = var.type
            if var_type is None:
                raise Exception(f"Could not find type for variable {var_name}")
            column = ast.ColumnDef(
                colname=var_name + "_scalar",
                typeName=ast.TypeName(names=(ast.String(str(var_type).lower()),)),
            )
            temp_table_cols.append(column)
        temp_table_cols.append(
            ast.ColumnDef(
                colname=TEMP_KEY_NAME,
                typeName=ast.TypeName(names=[ast.String("serial")]),
                constraints=(ast.Constraint(contype=ConstrType.CONSTR_PRIMARY),),
            )
        )
        create_table_stmt = ast.CreateStmt(
            relation=ast.RangeVar(
                relname=TEMP_TABLE_NAME,
                inh=True,
                relpersistence="t",
            ),
            tableElts=temp_table_cols,
            oncommit=OnCommitAction.ONCOMMIT_DROP,
        )
        self.schema.add_table_from_ast(create_table_stmt)
        block += (IndentedStream()(create_table_stmt) + ";").split("\n")

        # INSERT INTO temp SELECT * FROM UNNEST(param1, param2, ...);
        insert_stmt = ast.InsertStmt(
            relation=ast.RangeVar(relname="temp", inh=True, relpersistence="p"),
            selectStmt=ast.SelectStmt(
                targetList=[ast.A_Star()],
                fromClause=[
                    ast.FuncCall(
                        funcname=[ast.String("unnest")],
                        args=[
                            ast.ColumnRef(fields=[ast.String(param.name + "_batch")])
                            for param in self.params
                        ],
                    )
                ],
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
            f"FOR i IN ARRAY_LOWER({self.params[0].name}_batch, 1).."
            f"ARRAY_UPPER({self.params[0].name}_batch, 1) LOOP"
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
            block.append(
                lhs.name + "[i] := (" + self.rewrite_query_inside_loop(sql) + ");"
            )
        elif "PLpgSQL_stmt_execsql" in stmt:
            block.append(stmt["PLpgSQL_stmt_execsql"])
        elif "PLpgSQL_stmt_if" in stmt:
            sql = stmt["PLpgSQL_stmt_if"]["cond"]["PLpgSQL_expr"]["query"]
            block.append("IF (" + self.rewrite_query_inside_loop(sql) + ") THEN")
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
            block.append("IF returned[i] IS NULL THEN")
            block.append(
                [
                    "ret_vals[i] := (" + self.rewrite_query_inside_loop(sql) + ");",
                    "returned[i] := TRUE;",
                ]
            )
            block.append("END IF;")
        else:
            raise Exception("Unknown statement type: " + str(stmt))

    def put_stmt_toplevel(self, stmt, block):
        """
        TODO(kai): Should rewrite SELECT .. into x -> x := SELECT ..
        for performance reasons
        :param stmt:
        :param block:
        :return:
        """
        if "PLpgSQL_stmt_execsql" in stmt:
            substmt = stmt["PLpgSQL_stmt_execsql"]
            query = substmt["sqlstmt"]["PLpgSQL_expr"]["query"]
            query_ast = parse_sql(query)[0].stmt
            if self.is_batchable(stmt):
                local_vars = get_column_refs(query_ast).intersection(
                    self.local_var_names
                )
                local_vars_list = sorted(list(local_vars))
                for var_name in local_vars_list:
                    func_call_functions = (
                        (
                            ast.FuncCall(
                                funcname=(ast.String("unnest"),),
                                args=(ast.ColumnRef(fields=(ast.String("average"),)),),
                                agg_within_group=False,
                                agg_star=False,
                                agg_distinct=False,
                                func_variadic=False,
                            ),
                            None,
                        ),
                    )
                    update_stmt_ast = ast.UpdateStmt(
                        relation=ast.RangeVar(relname=TEMP_TABLE_NAME, inh=True),
                        targetList=[
                            ast.ResTarget(
                                name=var_name + "_scalar",
                                val=ast.ColumnRef(
                                    fields=[
                                        ast.String(var_name + "_var"),
                                    ]
                                ),
                            )
                        ],
                        whereClause=ast.A_Expr(
                            kind=A_Expr_Kind.AEXPR_OP,
                            name=(ast.String("="),),
                            lexpr=ast.ColumnRef(fields=[ast.String(var_name + "_key")]),
                            rexpr=ast.ColumnRef(
                                fields=[ast.String(TEMP_TABLE_NAME.lower() + "_key")]
                            ),
                        ),
                        fromClause=(
                            ast.RangeFunction(
                                lateral=False,
                                ordinality=True,
                                is_rowsfrom=False,
                                functions=func_call_functions,
                                alias=ast.Alias(
                                    aliasname=var_name + "_array",
                                    colnames=(
                                        ast.String(var_name + "_var"),
                                        ast.String(var_name + "_key"),
                                    ),
                                ),
                            ),
                        ),
                    )
                    update_stmt = IndentedStream()(update_stmt_ast) + ";"
                    block.append(update_stmt.split("\n"))
                if len(local_vars_list) > 0:
                    ColumnRefReplacer(
                        {
                            var_name.lower(): var_name.lower() + "_scalar"
                            for var_name in local_vars_list
                        }
                    )(query_ast)
                substmt["sqlstmt"]["PLpgSQL_expr"]["query"] = IndentedStream()(
                    query_ast
                )

                self.put_batched_sql(substmt, block)
            else:
                select_stmt = query_ast[0].stmt
                # print(query)
                assert select_stmt.intoClause is not None
                var_name = select_stmt.intoClause.rel.relname
                select_stmt.intoClause = None
                varno = self.get_varno(var_name)
                self.put_looped_stmt(
                    {
                        "PLpgSQL_stmt_assign": {
                            "varno": varno,
                            "expr": {
                                "PLpgSQL_expr": {"query": IndentedStream()(substmt)}
                            },
                        }
                    },
                    block,
                )
        elif "PLpgSQL_stmt_assign" in stmt:
            substmt = stmt["PLpgSQL_stmt_assign"]["expr"]["PLpgSQL_expr"]["query"]
            substmt = substmt[len("SELECT ") :]
            if self.is_batchable(stmt):
                batched_sql = self.batch_query(substmt).split("\n")
                varno = stmt["PLpgSQL_stmt_assign"]["varno"]
                batched_sql[0] = "(" + batched_sql[0]
                batched_sql[-1] = batched_sql[-1] + ");"
                batched_sql[0] = gen_assign_stmt(self.vars[varno].name, batched_sql[0])
                block += batched_sql
            else:
                self.put_looped_stmt(stmt, block)
        else:
            self.put_looped_stmt(stmt, block)

    def init_local_var_refs(self):
        block = self.tree["action"]["PLpgSQL_stmt_block"]["body"]
        for stmt in block:
            if self.is_batchable(stmt):
                if "PLpgSQL_stmt_execsql" in stmt:
                    query = stmt["PLpgSQL_stmt_execsql"]["sqlstmt"]["PLpgSQL_expr"][
                        "query"
                    ]
                elif "PLpgSQL_stmt_assign" in stmt:
                    query = stmt["PLpgSQL_stmt_assign"]["expr"]["PLpgSQL_expr"]["query"]
                else:
                    raise Exception("Unknown statement type: " + str(stmt))
                root_ast = parse_sql(query)
                self.local_var_refs.update(get_column_refs(root_ast[0].stmt))
        self.local_var_refs = set(var.lower() for var in self.local_var_refs)
        self.local_var_refs = self.local_var_refs.intersection(self.local_var_names)

    def is_batchable(self, stmt):
        if "PLpgSQL_stmt_execsql" in stmt:
            query = stmt["PLpgSQL_stmt_execsql"]["sqlstmt"]["PLpgSQL_expr"]["query"]
            tokens = scan(query)
            real_query = any(
                token.kind == "RESERVED_KEYWORD" and token.name.upper() == "FROM"
                for token in tokens
            )
            if real_query:
                return True
            else:
                return False
        elif "PLpgSQL_stmt_assign" in stmt:
            query = stmt["PLpgSQL_stmt_assign"]["expr"]["PLpgSQL_expr"]["query"]
            tokens = scan(query)
            real_query = any(
                token.kind == "RESERVED_KEYWORD" and token.name.upper() == "FROM"
                for token in tokens
            )
            query = query[len("SELECT ") :]
            if real_query:
                return True
            else:
                return False
        else:
            return False

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
        Batches a UDF query. For example, given the query "SELECT x FROM table" and the
        key "k", this function will return "SELECT ARRAY_AGG(x order by k) FROM
            (SELECT * FROM params, LATERAL (SELECT a FROM table) dt0) dt1".
        """
        select_stmt = parse_sql(query)[0].stmt
        assert isinstance(select_stmt, ast.SelectStmt)

        target_aggs = []
        num_targets = len(select_stmt.targetList)
        for i, target in enumerate(select_stmt.targetList):
            target_aggs.append(f"agg_{i}")
            target.name = target_aggs[i]

        subselect = ast.RangeSubselect(
            lateral=False,
            subquery=select_stmt,
            # alias=ast.Alias(f"agg_0")
            # TODO: fix hardcoded alias
        )
        sublink = ast.SubLink(
            subLinkType=SubLinkType.EXPR_SUBLINK,
            subLinkId=0,
            subselect=subselect,
        )
        sublink_target = ast.ResTarget(name="agg_0", val=sublink)
        temp_table = ast.RangeVar(relname="temp", inh=True)

        array_aggs = []
        for i in range(num_targets):
            func_call = ast.FuncCall(
                funcname=(ast.String("array_agg"),),
                args=(ast.RangeVar(relname=f"agg_{i}", inh=True),),
                agg_order=(
                    ast.SortBy(node=ast.ColumnRef(fields=(ast.String(TEMP_KEY_NAME),))),
                ),
            )
            array_agg = ast.ResTarget(val=func_call)
            array_aggs.append(array_agg)

        inner_select = ast.SelectStmt(
            targetList=(
                ast.ResTarget(val=ast.ColumnRef(fields=(ast.String(TEMP_KEY_NAME),))),
                sublink_target,
            ),
            fromClause=(temp_table,),
        )

        new_query = ast.SelectStmt(
            targetList=tuple(array_aggs),
            fromClause=(
                ast.RangeSubselect(
                    lateral=False, subquery=inner_select, alias=ast.Alias("dt0")
                ),
            ),
        )

        print("new_query: ", new_query)

        if into is not None:
            new_query.intoClause = self.generate_into_clause(into)

        new_query_str = IndentedStream()(new_query)

        if self.remove_laterals:
            planner = Planner(self.schema)
            new_query_str = planner.remove_laterals(new_query_str)
        return new_query_str

    def rewrite_query_inside_loop(self, query: str):
        tokens = scan(query)
        rhs_ast = parse_sql(query)
        VarToArrayRefRewriter(self.get_local_var_names())(rhs_ast)
        new_query = IndentedStream()(rhs_ast)
        """
        HACK: We need to be able to tell if an assignment is a real query i.e. one 
        that SELECTs from the database, or just a PL/pgSQL expression like 
        "a := b + 1". We do this by checking if there is a FROM clause in the query.
    
        If it ends up not being a "real" query, we have to do some post-processing
        to make sure deparsing it doesn't result in it being executed as a real 
        query, which is an issue because the deparser automatically adds SELECT to
        the beginning of the query.
    
        We don't want the deparser to add SELECT to the beginning of the query
        because this makes it much slower.
        """
        real_query = any(
            token.kind == "RESERVED_KEYWORD" and token.name.upper() == "FROM"
            for token in tokens
        )
        if not real_query:
            first_token = tokens[0]
            assert (
                first_token.kind == "RESERVED_KEYWORD"
                and first_token.name.upper() == "SELECT"
            )
            new_query = new_query[len("SELECT ") :]
        return new_query

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

        # TODO(kai): We don't want to accidentally overwrite an existing function,
        # but for now we'll just replace it.
        self.sql_tree.replace = True

        # Change the function name
        self.original_func_name = self.sql_tree.funcname[0].val
        self.sql_tree.funcname = [ast.String(self.sql_tree.funcname[0].val + "_batch")]

    def get_varno(self, var_name):
        for varno, var in self.vars.items():
            if var.name == var_name:
                return varno
        raise Exception(f"Could not find varno for {var_name}")
