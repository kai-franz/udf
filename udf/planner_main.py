import pydot
from pglast import parse_sql, ast
from pglast.enums import SetOperation, JoinType, BoolExprType
from pglast.stream import IndentedStream

from udf.planner import (
    Result,
    Node,
    SetOp,
    DependentJoin,
    Join,
    Filter,
    AggFinder,
    Agg,
    Project,
    OrderBy,
    Limit,
    TableScan,
)
from udf.schema import Schema


class Planner:
    def __init__(self, schema: Schema):
        self.schema = schema

    def remove_laterals(self, query_str: str) -> str:
        plan = self.plan_query(query_str)
        plan.remove_dependent_joins()
        graph = pydot.Dot(graph_type="digraph")
        plan.visualize(graph)
        graph.write_png("plan_flattened.png")
        deparsed_ast = plan.deparse()
        return IndentedStream()(deparsed_ast)

    def plan_query(self, query_str: str):
        select_stmt = parse_sql(query_str)[0].stmt

        # Recursively plan the query
        result = Result(
            self.schema, self.plan_select(select_stmt), into=select_stmt.intoClause
        )

        graph = pydot.Dot(graph_type="digraph")
        result.visualize(graph)
        graph.write_png("plan_raw.png")

        # Push down filters
        result.push_down_filters()

        # Visualization
        graph = pydot.Dot(graph_type="digraph")
        result.visualize(graph)
        graph.write_png("plan.png")

        # deparsed_ast = result.deparse()
        # print(deparsed_ast)
        # print(IndentedStream()(deparsed_ast))
        return result

    def plan_select(self, select_stmt: ast.SelectStmt) -> Node:
        if select_stmt.op != SetOperation.SETOP_NONE:
            left = self.plan_select(select_stmt.larg)
            right = self.plan_select(select_stmt.rarg)
            return SetOp(self.schema, select_stmt, left, right)
        # 1. FROM
        if select_stmt.fromClause is None:
            from_tables = []
        else:
            from_tables = [
                self.plan_from_node(from_entry) for from_entry in select_stmt.fromClause
            ]

        # JOIN
        if len(from_tables) > 0:
            left_node = from_tables[0]
            join_tables = list(zip(from_tables, select_stmt.fromClause))[1:]
            for right_node, right_ast in join_tables:
                if isinstance(right_ast, ast.RangeSubselect) and right_ast.lateral:
                    left_node = DependentJoin(left_node, right_node, self.schema)
                else:  # assume cross join, filter handled in WHERE clause
                    left_node = Join(
                        self.schema,
                        right_ast,
                        left_node,
                        right_node,
                        join_type=JoinType.JOIN_INNER,
                    )
            node = left_node
        else:
            # No FROM clause
            node = None

        # WHERE

        if select_stmt.whereClause is not None:
            if isinstance(select_stmt.whereClause, ast.BoolExpr):
                assert select_stmt.whereClause.boolop == BoolExprType.AND_EXPR
                for qual in select_stmt.whereClause.args:
                    node = Filter(self.schema, qual, node)
            else:
                assert isinstance(select_stmt.whereClause, ast.A_Expr)
                node = Filter(self.schema, select_stmt.whereClause, node)

        # GROUP BY / Scalar aggregate
        agg_targets = []
        for target in select_stmt.targetList:
            agg_finder = AggFinder()
            agg_finder(target)
            agg_targets.append(agg_finder.has_agg)

        if select_stmt.groupClause is not None or any(agg_targets):
            node = Agg(
                self.schema,
                select_stmt.targetList,
                select_stmt.groupClause,
                agg_targets,
                node,
            )
        else:
            # SELECT
            assert (
                select_stmt.targetList is not None and len(select_stmt.targetList) > 0
            )
            node = Project(self.schema, select_stmt.targetList, node)

        # ORDER BY
        if select_stmt.sortClause is not None:
            node = OrderBy(self.schema, select_stmt.sortClause, node)

        # LIMIT
        if select_stmt.limitCount is not None:
            node = Limit(self.schema, select_stmt, node)

        return node

    def plan_from_node(self, from_node):
        if isinstance(from_node, ast.RangeVar):
            return TableScan(self.schema, from_node)
        elif isinstance(from_node, ast.RangeSubselect):
            return self.plan_select(from_node.subquery)
        elif isinstance(from_node, ast.JoinExpr):
            left = self.plan_from_node(from_node.larg)
            right = self.plan_from_node(from_node.rarg)
            return Join(self.schema, from_node, left, right)
        else:
            raise NotImplementedError
