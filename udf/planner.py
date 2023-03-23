from collections import defaultdict
from enum import Enum

from pglast import ast, parse_sql
import pydot
from pglast.enums import JoinType

AGG_FUNCS = {
    "array_agg",
    "avg",
    "bit_and",
    "bit_or",
    "bool_and",
    "bool_or",
    "count",
    "every",
    "json_agg",
    "jsonb_agg",
    "json_object_agg",
    "max",
    "min",
    "string_agg",
    "sum",
    "xmlagg",
}


class NodeType(Enum):
    TABLE_SCAN = 0
    FILTER = 1
    AGG = 2
    HAVING = 3
    PROJECT = 4
    JOIN = 5
    DEPENDENT_JOIN = 6
    ORDER_BY = 7
    RESULT = 8

    def camel_case_name(self):
        return "".join([w.capitalize() for w in self.name.split("_")])


class Ordering:
    TABLE_SCAN = 0
    JOIN = 0
    FILTER = 1
    AGG = 2
    HAVING = 3
    PROJECT = 4
    ORDER_BY = 5

    @staticmethod
    def can_coalesce(parent_order: int, child_order: int):
        return parent_order <= child_order

    @staticmethod
    def get_order(node_type: NodeType):
        return getattr(Ordering, node_type.name)


class Node:
    next_ids = defaultdict(int)

    def __init__(self, type: NodeType):
        self.children = []
        if type != NodeType.TABLE_SCAN:
            self.id = Node.next_id(Node.next_ids[type.name])
            Node.next_ids[type.name] += 1
            self.graph_node = pydot.Node(
                type.name + "_" + str(self.id), label=type.camel_case_name()
            )

    @staticmethod
    def next_id(node_type):
        Node.next_ids[node_type] += 1
        return Node.next_ids[node_type]

    def visualize(self, graph: pydot.Dot):
        graph.add_node(self.graph_node)
        for child in self.children:
            graph.add_edge(pydot.Edge(self.graph_node, child.visualize(graph)))
        return self.graph_node

    def child(self):
        return self.children[0]

    def left(self):
        return self.children[0]

    def right(self):
        return self.children[1]


class TableScan(Node):
    def __init__(self, ast_node: ast.RangeVar):
        super().__init__(NodeType.TABLE_SCAN)
        self.ast_node = ast_node
        self.table = ast_node.relname
        self.graph_node = pydot.Node(self.table)

    def deparse(self):
        return self.ast_node


class Filter(Node):
    def __init__(self, predicate, child):
        super().__init__(type=NodeType.FILTER)
        self.predicate = predicate
        self.children.append(child)


class Project(Node):
    def __init__(self, exprs, child):
        super().__init__(type=NodeType.PROJECT)
        self.exprs = exprs
        self.children.append(child)


class Join(Node):
    def __init__(self, ast_node: ast.JoinExpr, left, right, join_type=None):
        super().__init__(type=NodeType.JOIN)
        self.ast_node = ast_node
        self.children.append(left)
        self.children.append(right)
        if join_type is None:
            self.join_type = JoinType.JOIN_INNER
        else:
            self.join_type = ast_node.jointype

    def deparse(self):
        return ast.JoinExpr(
            jointype=self.join_type,
            larg=self.left().deparse(),
            rarg=self.right().deparse(),
            isNatural=False,
            quals=self.ast_node.quals,
        )


class DependentJoin(Node):
    def __init__(self, left, right):
        super().__init__(type=NodeType.DEPENDENT_JOIN)
        self.children.append(left)
        self.children.append(right)


class Agg(Node):
    def __init__(self, exprs, child):
        super().__init__(type=NodeType.AGG)
        self.exprs = exprs
        self.children.append(child)


class OrderBy(Node):
    def __init__(self, sort_clause, child):
        super().__init__(type=NodeType.ORDER_BY)
        self.sort_clause = sort_clause
        self.children.append(child)


class Result(Node):
    def __init__(self, child):
        super().__init__(type=NodeType.RESULT)
        self.children.append(child)


class Planner:
    def __init__(self):
        pass

    @staticmethod
    def plan_query(query_str: str):
        select_stmt = parse_sql(query_str)[0].stmt
        print(select_stmt)
        result = Result(Planner.plan_select(select_stmt))

        # Visualization
        graph = pydot.Dot(graph_type="digraph")
        result.visualize(graph)
        graph.write_png("plan.png")

    @staticmethod
    def plan_select(select_stmt: ast.SelectStmt):
        # 1. FROM
        if select_stmt.fromClause is None:
            from_tables = []
        else:
            from_tables = [
                Planner.plan_from_node(from_entry)
                for from_entry in select_stmt.fromClause
            ]

        # JOIN
        left_node = from_tables[0]
        for right_node, right_ast in list(zip(from_tables, select_stmt.fromClause))[1:]:
            if isinstance(right_ast, ast.RangeSubselect) and right_ast.lateral:
                left_node = DependentJoin(left_node, right_node)
            else:  # assume cross join, filter handled in WHERE
                left_node = Join(right_ast, left_node, right_node, join_type="cross")
        node = left_node

        # WHERE
        if select_stmt.whereClause is not None:
            node = Filter(select_stmt.whereClause, node)

        # GROUP BY / Scalar aggregate
        if select_stmt.groupClause is not None:
            node = Agg(select_stmt.groupClause, node)

        # SELECT
        assert select_stmt.targetList is not None and len(select_stmt.targetList) > 0
        node = Project(select_stmt.targetList, node)

        # ORDER BY
        if select_stmt.sortClause is not None:
            node = OrderBy(select_stmt.sortClause, node)

        return node

    @staticmethod
    def plan_from_node(from_node):
        if isinstance(from_node, ast.RangeVar):
            return TableScan(from_node)
        elif isinstance(from_node, ast.RangeSubselect):
            return Planner.plan_select(from_node.subquery)
        elif isinstance(from_node, ast.JoinExpr):
            left = Planner.plan_from_node(from_node.larg)
            right = Planner.plan_from_node(from_node.rarg)
            return Join(from_node, left, right)
        else:
            raise NotImplementedError


if __name__ == "__main__":
    #     Planner.plan_query(
    #         """SELECT ca_state
    #      , d_year
    #      , d_qoy
    #      , totallargepurchases(ca_state, 1000, d_year, d_qoy)
    # FROM customer_address, date_dim
    # WHERE d_year IN (1998, 1999, 2000)
    #   AND ca_state IS NOT NULL
    # GROUP BY ca_state, d_year, d_qoy
    # ORDER BY ca_state
    #        , d_year
    #        , d_qoy;"""
    #     )

    Planner.plan_query(
        """select * from customer inner join web_sales on customer.c_customer_sk = web_sales.ws_sold_date_sk;"""
    )
