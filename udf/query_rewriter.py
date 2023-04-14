import copy
from typing import List
from pglast import ast
from pglast.enums import SortByNulls, SortByDir
from pglast.stream import IndentedStream
from pglast.visitors import Visitor


def convert_function_header(f: ast.CreateFunctionStmt):
    name = f.funcname[0].val
    argsTypeName = ast.String(name + "args")
    argType = ast.TypeName([argsTypeName], arrayBounds=[ast.Integer(-1)])
    f.parameters = [ast.FunctionParameter("argarray", argType=argType)]

    # convert return statement to a table
    f.returnType.setof = True
    returnTypeList = list(f.returnType.names)
    returnTypeList[1] = ast.String("record")
    f.returnType.names = tuple(returnTypeList)


def get_function_calls(parseTree, udf_name):
    """
    Removes all function calls from the query,
    returning the list of function calls removed.
    :param q: the query to modify
    :return: a list of function calls removed
    """

    class FunctionCallAccumulator(Visitor):
        function_calls = []

        def visit_FuncCall(self, parent, node):
            print(node.funcname)
            if node.funcname[-1].val == udf_name:
                self.function_calls.append(node)

    visitor = FunctionCallAccumulator()
    visitor(parseTree)
    return visitor.function_calls


class FunctionCallSplitter(Visitor):
    """
    Given an expression like f(x + 2) > 0, we want to split this expression into two
    parts:
    1. f(x + 2)
    2. f > 0
    """

    def __init__(self, func_name: str):
        self.func_name = func_name

    def visit_FuncCall(self, parent, node):
        if node.funcname[-1].val == self.func_name:
            return ast.ColumnRef(fields=(self.func_name,))


def convert_to_subquery(q: ast.SelectStmt) -> ast.SelectStmt:
    """
    Converts a query of the form (SELECT c1, c2, c3 FROM ...) into
    (SELECT c1, c2, c3 FROM (SELECT c1, c2, c3 FROM ...) as dt1)
    :param q: the query to convert
    :return: the modified query, as a SelectStmt
    """
    from_star = ast.ColumnRef((ast.A_Star(),))
    from_target = ast.ResTarget(val=from_star)
    subselect = ast.RangeSubselect(lateral=False, subquery=q, alias=ast.Alias("dt1"))
    outer_target_list = q.targetList
    q.targetList = (from_target,)
    new_query = ast.SelectStmt(targetList=outer_target_list, fromClause=(subselect,))
    return new_query


def get_array_agg(q: ast.SelectStmt, col_refs: List[ast.ColumnRef]):
    """
    Converts a query of the form (SELECT c1, c2, c3 FROM ...) into
    (SELECT array_agg(row(c1, c2, c3)) FROM ...)
    using col_refs as the target list for the array_agg
    :param q:
    :param col_refs:
    :return:
    """
    func_call = ast.FuncCall(funcname=(ast.String("array_agg"),), args=tuple(col_refs))
    return ast.ResTarget(name="batch", val=func_call)


def new_array_agg(col_refs: List):
    """
    Returns an array_agg function call with the given column references
    :param col_refs:
    :return:
    """
    sort_by = []
    for col_ref in col_refs:
        sort_by.append(
            ast.SortBy(
                node=col_ref,
                sortby_dir=SortByDir.SORTBY_DEFAULT,
                sortby_nulls=SortByNulls.SORTBY_NULLS_DEFAULT,
            )
        )

    targetlist = []
    for col_ref in col_refs:
        if isinstance(col_ref, ast.ColumnRef):
            batched_alias = col_ref.fields[0].val + "_batch"
        elif isinstance(col_ref, ast.A_Const):
            batched_alias = str(col_ref.val.val) + "_batch"
        else:
            batched_alias = "query_" + str(hash(str(col_ref))) + "_batch"
        func_call = ast.FuncCall(
            funcname=(ast.String("array_agg"),), args=[col_ref], agg_order=sort_by
        )
        targetlist.append(ast.ResTarget(name=batched_alias, val=func_call))
    return targetlist


def get_unique_col_refs(col_refs: List[ast.ColumnRef]):
    """
    Returns a list of unique column references
    :param col_refs:
    :return:
    """

    # TODO(kai): for now we don't worry about deduplication
    # col_ref_strs = set()
    # unique_col_refs = []
    # for col_ref in col_refs:
    #     if col_ref.fields[0].val not in col_ref_strs:
    #         unique_col_refs.append(col_ref)
    #         col_ref_strs.add(col_ref.fields[0].val)
    return col_refs


def remove_where_clause_udf(select_stmt, udf_name):
    where_clause = select_stmt.whereClause
    udf_clause = None
    if isinstance(where_clause, ast.A_Expr):
        udf_clause = where_clause
        select_stmt.whereClause = None
    elif isinstance(where_clause, ast.BoolExpr):
        for i, clause in enumerate(where_clause.args):
            fn_calls = get_function_calls(clause, udf_name)
            if len(fn_calls) > 0:
                udf_clause = clause
                # remove the ith element from where_clause
                where_clause.args = where_clause.args[:i] + where_clause.args[i + 1 :]
                break
    else:
        raise Exception("Unexpected where clause type")
    if not udf_clause:
        raise Exception("UDF not found in where clause")
    return udf_clause


def transform_query(q, udf_name):
    """
    Transforms a query into a batched form
    :param q:
    :return:
    """
    print(udf_name)
    subquery = q[0].stmt
    q[0].stmt = convert_to_subquery(subquery)
    select_stmt = q[0].stmt
    fn_calls = []
    udf_in_where_clause = False
    udf_clause = None
    if subquery.whereClause is not None:
        fn_calls = get_function_calls(subquery.whereClause, udf_name)
        if len(fn_calls) > 0:
            assert len(fn_calls) == 1
            print("UDF in where clause")
            # UDF in the where clause. We need to move it to the select clause.
            # Then, at the end, we'll add the where clause back in.
            udf_in_where_clause = True
            udf_clause = remove_where_clause_udf(subquery, udf_name)
            select_stmt.targetList = select_stmt.targetList + (
                ast.ResTarget(val=fn_calls[0]),
            )
            fn_calls[0] = select_stmt.targetList[-1]
            print(select_stmt.targetList)
            FunctionCallSplitter(udf_name)(udf_clause)
            print("udf clause:", udf_clause)

    print(select_stmt.targetList)
    fn_calls = get_function_calls(q, udf_name)
    if len(fn_calls) == 0:
        # No UDFs. Just return.
        raise Exception("No UDFs found")
        # return
    assert len(fn_calls) == 1

    inner_targets = []
    fn_call = fn_calls[0]
    fn_call_args = fn_call.args
    print(select_stmt)

    # for target in select_stmt.targetList:
    #     if not (
    #         isinstance(target.val, ast.FuncCall)
    #         and target.val.funcname[0].val == udf_name
    #     ):
    #         col_refs.append(target.val)
    # print(inner_targets)
    # col_refs = getUniqueColRefs(col_refs)
    unnest_target_list = copy.deepcopy(select_stmt.targetList)
    select_stmt.targetList = new_array_agg(list(inner_targets) + list(fn_call_args))

    # TODO(kai): make sure these are all the columns we need from the inner query
    inner_target_list = [
        target
        for target in copy.deepcopy(unnest_target_list)
        if not (
            isinstance(target.val, ast.FuncCall) and target.val.funcname[-1] == udf_name
        )
    ]
    # Construct unnest_target_list, which is the target list for the outer query.
    # This consists of unnesting columns that we were previously selecting,
    # as well as unnesting the function call result.
    outer_target_list_for_where_clause = []
    for target in unnest_target_list:
        val = target.val
        alias = None
        # The UDF call. Change this to unnest the batched version of the function.
        if isinstance(val, ast.FuncCall):
            alias = val.funcname[0].val
            val.funcname = [ast.String(val.funcname[0].val + "_batch")]
            new_args = []
            for arg in val.args:
                if isinstance(arg, ast.FuncCall) and arg.funcname[0].val == udf_name:
                    continue
                if isinstance(arg, ast.ColumnRef):
                    new_args.append(
                        ast.ColumnRef([ast.String(arg.fields[0].val + "_batch")])
                    )
                elif isinstance(arg, ast.A_Const):
                    new_args.append(ast.ColumnRef([str(arg.val.val) + "_batch"]))
                else:
                    new_args.append(
                        ast.ColumnRef(["query_" + str(hash(str(arg))) + "_batch"])
                    )

            val.args = new_args
            unnest_call = ast.FuncCall(funcname=[ast.String("unnest")], args=[val])
            target.val = unnest_call
            target.name = alias
        if isinstance(val, ast.ColumnRef):
            alias = val.fields[-1].val
            val.fields = [
                ast.FuncCall(
                    funcname=[ast.String("unnest")],
                    args=[ast.ColumnRef([ast.String(val.fields[-1].val + "_batch")])],
                )
            ]
            target.name = alias
        if isinstance(val, ast.A_Const):
            target.val = ast.FuncCall(
                funcname=[ast.String("unnest")],
                args=[ast.ColumnRef(["_batch"])],
            )
            alias = "unknown"
            raise Exception("Not implemented")
        if alias is not None:
            outer_target_list_for_where_clause.append(
                ast.ResTarget(val=ast.ColumnRef([alias]))
            )
    subquery.targetList = inner_target_list
    # push down the select statement into a subquery
    subselect = ast.RangeSubselect(
        lateral=False, subquery=select_stmt, alias=ast.Alias("dt2")
    )
    select_stmt = ast.SelectStmt(targetList=unnest_target_list, fromClause=(subselect,))
    select_stmt.targetList = unnest_target_list

    if udf_in_where_clause:
        outer_select_stmt = ast.SelectStmt(
            targetList=outer_target_list_for_where_clause,
            fromClause=(
                ast.RangeSubselect(
                    lateral=False, subquery=select_stmt, alias=ast.Alias("dt3")
                ),
            ),
        )
        outer_select_stmt.whereClause = udf_clause
        q[0].stmt = outer_select_stmt
    else:
        q[0].stmt = select_stmt
