from typing import List

from pglast import *
import pglast
import pprint
from pglast import enums, ast
from pglast.stream import IndentedStream, RawStream
from pglast.visitors import Visitor, Delete, Skip


def convertFunctionHeader(f: ast.CreateFunctionStmt):
    name = f.funcname[0].val
    argsTypeName = ast.String(name + "args")
    argType = ast.TypeName([argsTypeName], arrayBounds=[ast.Integer(-1)])
    f.parameters = [ast.FunctionParameter("argarray", argType=argType)]

    # convert return statement to a table
    # retTypeName =
    f.returnType.setof = True
    returnTypeList = list(f.returnType.names)
    returnTypeList[1] = ast.String("record")
    f.returnType.names = tuple(returnTypeList)


def removeFunctionCalls(parseTree):
    """
    Removes all function calls from the query,
    returning the list of function calls removed.
    :param q: the query to modify
    :return: a list of function calls removed
    """

    class FunctionCallRemover(Visitor):
        function_calls = []

        def visit_SelectStmt(self, parent, node):
            non_function_calls = []
            if node.targetList is None:
                return
            for target in node.targetList:
                if isinstance(target.val, ast.FuncCall):
                    self.function_calls.append(target.val)
                else:
                    non_function_calls.append(target)
            node.targetList = tuple(non_function_calls)

    visitor = FunctionCallRemover()
    visitor(parseTree)
    return visitor.function_calls


def convertToSubquery(q: ast.SelectStmt):
    """
    Converts a query of the form (SELECT c1, c2, c3, ...) into
    (SELECT c1, c2, c3 FROM (SELECT * ...) as dt1)
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


def getArrayAgg(q: ast.SelectStmt, col_refs: List[ast.ColumnRef]):
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


def getUniqueColRefs(col_refs: List[ast.ColumnRef]):
    """
    Returns a list of unique column references
    :param col_refs:
    :return:
    """
    col_ref_strs = set()
    unique_col_refs = []
    for col_ref in col_refs:
        if col_ref.fields[0].sval not in col_ref_strs:
            unique_col_refs.append(col_ref)
            col_ref_strs.add(col_ref.fields[0].sval)
    return unique_col_refs


def transformQuery(q):
    """
    Transforms a query into a batched form
    :param q:
    :return:
    """
    q[0].stmt = convertToSubquery(q[0].stmt)
    fn_calls = removeFunctionCalls(q)
    col_refs = []
    assert (len(fn_calls) == 1)
    fn_call = fn_calls[0]
    for arg in fn_call.args:
        if isinstance(arg, ast.ColumnRef):
            col_refs.append(arg)

    for target in q[0].stmt.targetList:
        if isinstance(target, ast.ResTarget):
            if isinstance(target.val, ast.ColumnRef):
                col_refs.append(target.val)

    # get
    col_refs = getUniqueColRefs(col_refs)
    q[0].stmt.targetList = (getArrayAgg(q[0].stmt, col_refs),)

    # push down q[0].stmt into a subquery
    outer_target_list = (fn_call,)
    subselect = ast.RangeSubselect(lateral=False, subquery=q[0].stmt, alias=ast.Alias("dt2"))
    q[0].stmt = ast.SelectStmt(targetList=outer_target_list, fromClause=(subselect,))

    batched_func_call = ast.FuncCall(funcname=(ast.String(fn_call.funcname[0].sval + "_batched"),),
                                     args=(ast.ColumnRef((ast.String("batch"),)),))
    indirection_target = ast.A_Indirection(arg=batched_func_call, indirection=(ast.A_Star(),))
    q[0].stmt.targetList = (ast.ResTarget(val=indirection_target),)

    print("unique column refs:", col_refs)
