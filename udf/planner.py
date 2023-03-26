from collections import defaultdict
from enum import Enum, IntEnum, auto
from pglast import ast, parse_sql
from pglast.enums import JoinType, SetOperation
from pglast.stream import IndentedStream
import pydot
from pglast.visitors import Visitor

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


class AggFinder(Visitor):
    def __init__(self):
        self.has_agg = False

    def visit_FuncCall(self, parent, node: ast.FuncCall):
        if node.funcname and node.funcname[-1].val in AGG_FUNCS:
            self.has_agg = True


class NodeType(Enum):
    TABLE_SCAN = 0
    UNION = auto()
    INTERSECT = auto()
    EXCEPT = auto()
    FILTER = auto()
    AGG = auto()
    HAVING = auto()
    PROJECT = auto()
    JOIN = auto()
    DEPENDENT_JOIN = auto()
    ORDER_BY = auto()
    LIMIT = auto()
    RESULT = auto()

    def camel_case_name(self):
        return "".join([w.capitalize() for w in self.name.split("_")])

    @staticmethod
    def from_setop(set_op: SetOperation):
        if set_op == SetOperation.SETOP_UNION:
            return NodeType.UNION
        elif set_op == SetOperation.SETOP_INTERSECT:
            return NodeType.INTERSECT
        elif set_op == SetOperation.SETOP_EXCEPT:
            return NodeType.EXCEPT
        raise Exception("Unknown set operation")

    def to_setop(self):
        if self == NodeType.UNION:
            return SetOperation.SETOP_UNION
        elif self == NodeType.INTERSECT:
            return SetOperation.SETOP_INTERSECT
        elif self == NodeType.EXCEPT:
            return SetOperation.SETOP_EXCEPT
        raise Exception("Unknown set operation")


class Ordering:
    TABLE_SCAN = 0
    JOIN = 0
    UNION = 1
    INTERSECT = 1
    EXCEPT = 1
    FILTER = 2
    AGG = 3
    HAVING = 4
    PROJECT = 5
    ORDER_BY = 6
    LIMIT = 7

    @staticmethod
    def can_coalesce(parent_order: int, child_order: int):
        return parent_order > child_order or (
            parent_order == child_order and parent_order != Ordering.AGG
        )


class Node:
    ids = defaultdict(int)
    derived_table_count = 0

    def __init__(self, type: NodeType):
        self.children = []
        self.type = type
        if type != NodeType.TABLE_SCAN:
            self.id = Node.next_id(type.name)
            Node.ids[type.name] += 1
            self.graph_node = pydot.Node(
                type.name + "_" + str(self.id), label=type.camel_case_name()
            )

    @staticmethod
    def next_id(node_type):
        Node.ids[node_type] += 1
        return Node.ids[node_type]

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

    @staticmethod
    def next_derived_table():
        Node.derived_table_count += 1
        return f"t{Node.derived_table_count}"

    def construct_subselect(self, child_ast) -> ast.SelectStmt:
        """
        Takes in an AST, and constructs a SelectStmt that wraps it.
        If the AST is already a SelectStmt, it will be returned as-is,
        unless it is a set operation, in which case it will still be
        wrapped in a SelectStmt.
        :param child_ast:
        :return:
        """
        if (
            isinstance(child_ast, ast.SelectStmt)
            and child_ast.op == SetOperation.SETOP_NONE
        ):
            # If the child is already a non-setop select statement, we can just return it.
            return child_ast
        if isinstance(child_ast, ast.SelectStmt):
            # Handle the case where the child is a set operation.
            # Here, we need to wrap the child in a RangeSubselect, since it's going to
            # be in the FROM clause.
            child_ast = ast.RangeSubselect(
                lateral=False,
                subquery=child_ast,
                alias=ast.Alias(Node.next_derived_table()),
            )
        return ast.SelectStmt(
            targetList=[],
            op=SetOperation.SETOP_NONE,
            fromClause=[child_ast],
        )

    def construct_subselect_if_needed(self, child_ast) -> ast.SelectStmt:
        """
        Takes in an AST, and constructs a SelectStmt that wraps it.
        If the AST is already a SelectStmt, it will be returned as-is,
        even if it is a set operation.
        :param child_ast:
        :return:
        """
        if isinstance(child_ast, ast.SelectStmt):
            # If the child is already a select statement, we can just return it.
            return child_ast
        return ast.SelectStmt(
            targetList=[],
            op=SetOperation.SETOP_NONE,
            fromClause=[child_ast],
        )

    def force_construct_subselect(self, child_ast) -> ast.SelectStmt:
        if isinstance(child_ast, ast.SelectStmt):
            # Handle the case where the child is a set operation.
            # Here, we need to wrap the child in a RangeSubselect, since it's going to
            # be in the FROM clause.
            child_ast = ast.RangeSubselect(
                lateral=False,
                subquery=child_ast,
                alias=ast.Alias(Node.next_derived_table()),
            )
        return ast.SelectStmt(
            targetList=[],
            op=SetOperation.SETOP_NONE,
            fromClause=[child_ast],
        )

    def construct_range_subselect(self, child_ast) -> ast.RangeSubselect:
        subquery = self.construct_subselect(child_ast)
        if subquery.targetList is not None and len(subquery.targetList) == 0:
            subquery.targetList = None
        return ast.RangeSubselect(
            lateral=False,
            subquery=subquery,
            alias=ast.Alias(Node.next_derived_table()),
        )

    def rewrite_child_if_needed(self, child_ast, child_order: int):
        parent_order = self.get_order()
        if Ordering.can_coalesce(parent_order, child_order):
            return child_ast
        else:
            if not isinstance(child_ast, ast.SelectStmt):
                child_ast = self.construct_subselect(child_ast)
                child_ast.targetList = [
                    ast.ResTarget(val=ast.ColumnRef((ast.A_Star(),)))
                ]
            assert child_ast.targetList is None or len(child_ast.targetList) > 0
            return ast.SelectStmt(
                targetList=[],
                op=SetOperation.SETOP_NONE,
                fromClause=[
                    ast.RangeSubselect(
                        lateral=False,
                        subquery=child_ast,
                        alias=ast.Alias(Node.next_derived_table()),
                    )
                ],
            )

    def get_order(self):
        return getattr(Ordering, self.type.name)


class TableScan(Node):
    def __init__(self, ast_node: ast.RangeVar):
        super().__init__(NodeType.TABLE_SCAN)
        self.ast_node = ast_node
        self.table = ast_node.relname
        self.graph_node = pydot.Node(self.table)

    def deparse(self):
        return self.ast_node


class SetOp(Node):
    def __init__(self, ast_node: ast.SelectStmt, left: Node, right: Node):
        super().__init__(NodeType.from_setop(ast_node.op))
        self.ast_node = ast_node
        self.all = ast_node.all
        self.children.append(left)
        self.children.append(right)

    def deparse(self):
        child_asts = [child.deparse() for child in self.children]
        child_asts = [self.construct_subselect_if_needed(ast) for ast in child_asts]
        left_ast = child_asts[0]
        right_ast = child_asts[1]
        return ast.SelectStmt(
            op=self.type.to_setop(),
            all=self.all,
            larg=left_ast,
            rarg=right_ast,
            targetList=None,
            fromClause=None,
        )


class Filter(Node):
    def __init__(self, predicate, child):
        super().__init__(type=NodeType.FILTER)
        self.predicate = predicate
        self.children.append(child)

    def deparse(self):
        child_ast = self.rewrite_child_if_needed(
            self.child().deparse(), self.child().get_order()
        )
        child_ast = self.construct_subselect(child_ast)

        if child_ast.whereClause is None:
            child_ast.whereClause = []
        child_ast.whereClause = self.predicate
        return child_ast


class Project(Node):
    def __init__(self, exprs, child):
        super().__init__(type=NodeType.PROJECT)
        self.exprs = exprs
        if child is not None:
            self.children.append(child)

    def deparse(self):
        if len(self.children) == 0:
            # SELECT without FROM clause
            return ast.SelectStmt(
                targetList=self.exprs,
                op=SetOperation.SETOP_NONE,
                fromClause=None,
            )
        child_ast = self.child().deparse()
        print("child_ast: ", child_ast)
        # For now, I've decided to always wrap the child in a subselect
        # for project nodes.
        child_ast = self.construct_subselect(child_ast)
        if (
            child_ast.targetList is None
            and len(child_ast.targetList) > 0
            or self.child().get_order() == Ordering.PROJECT
        ):
            child_ast = self.force_construct_subselect(child_ast)
        child_ast.targetList = list(child_ast.targetList) + list(self.exprs)
        return child_ast


class Join(Node):
    def __init__(self, ast_node: ast.JoinExpr, left, right, join_type=None):
        super().__init__(type=NodeType.JOIN)
        self.ast_node = ast_node
        self.children.append(left)
        self.children.append(right)
        if join_type is None:
            # Explicit join; infer join type from AST
            self.join_type = ast_node.jointype
            self.quals = ast_node.quals
        else:
            # Implicit join; use join type provided by caller
            self.join_type = JoinType.JOIN_INNER
            self.quals = None

    def deparse(self):
        return ast.JoinExpr(
            jointype=self.join_type,
            larg=self.left().deparse(),
            rarg=self.right().deparse(),
            isNatural=False,
            quals=self.quals,
        )


class DependentJoin(Node):
    def __init__(self, left, right):
        super().__init__(type=NodeType.DEPENDENT_JOIN)
        self.children.append(left)
        self.children.append(right)


class Agg(Node):
    def __init__(self, target_list, group_clause, agg_targets, child):
        super().__init__(type=NodeType.AGG)
        self.group_clause = group_clause
        self.agg_keys = [
            target for target, has_agg in zip(target_list, agg_targets) if not has_agg
        ]
        self.agg_values = [
            target for target, has_agg in zip(target_list, agg_targets) if has_agg
        ]
        if len(self.agg_keys) == 0:
            assert group_clause is None
            self.agg_keys = None
        self.target_list = target_list
        self.children.append(child)

    def deparse(self):
        parent_ast = self.rewrite_child_if_needed(
            self.child().deparse(), self.child().get_order()
        )
        parent_ast = self.construct_subselect(parent_ast)
        parent_ast.groupClause = self.agg_keys
        parent_ast.targetList = self.target_list
        return parent_ast


class OrderBy(Node):
    def __init__(self, sort_clause, child):
        super().__init__(type=NodeType.ORDER_BY)
        self.sort_clause = sort_clause
        self.children.append(child)

    def deparse(self):
        parent_ast = self.rewrite_child_if_needed(
            self.child().deparse(), self.child().get_order()
        )
        parent_ast = self.construct_subselect(parent_ast)
        parent_ast.sortClause = self.sort_clause
        return parent_ast


class Limit(Node):
    def __init__(self, select_stmt, child):
        super().__init__(type=NodeType.LIMIT)
        self.limit_count = select_stmt.limitCount
        self.limit_offset = select_stmt.limitOffset
        self.limit_option = select_stmt.limitOption
        self.children.append(child)

    def deparse(self):
        parent_ast = self.rewrite_child_if_needed(
            self.child().deparse(), self.child().get_order()
        )
        parent_ast = self.construct_subselect(parent_ast)
        parent_ast.limitCount = self.limit_count
        parent_ast.limitOffset = self.limit_offset
        parent_ast.limitOption = self.limit_option
        return parent_ast


class Result(Node):
    def __init__(self, child):
        super().__init__(type=NodeType.RESULT)
        self.children.append(child)

    def deparse(self):
        ast = self.child().deparse()
        # print(ast)
        return ast


class Planner:
    def __init__(self):
        pass

    @staticmethod
    def plan_query(query_str: str):
        select_stmt = parse_sql(query_str)[0].stmt
        result = Result(Planner.plan_select(select_stmt))

        # Visualization
        graph = pydot.Dot(graph_type="digraph")
        result.visualize(graph)
        graph.write_png("plan.png")
        return result

    @staticmethod
    def plan_select(select_stmt: ast.SelectStmt) -> Node:
        if select_stmt.op != SetOperation.SETOP_NONE:
            left = Planner.plan_select(select_stmt.larg)
            right = Planner.plan_select(select_stmt.rarg)
            return SetOp(select_stmt, left, right)
        # 1. FROM
        if select_stmt.fromClause is None:
            from_tables = []
        else:
            from_tables = [
                Planner.plan_from_node(from_entry)
                for from_entry in select_stmt.fromClause
            ]

        # JOIN
        if len(from_tables) > 0:
            left_node = from_tables[0]
            for right_node, right_ast in list(zip(from_tables, select_stmt.fromClause))[
                1:
            ]:
                if isinstance(right_ast, ast.RangeSubselect) and right_ast.lateral:
                    left_node = DependentJoin(left_node, right_node)
                else:  # assume cross join, filter handled in WHERE clause
                    left_node = Join(
                        right_ast, left_node, right_node, join_type=JoinType.JOIN_INNER
                    )
            node = left_node
        else:
            # No FROM clause
            node = None

        # WHERE
        if select_stmt.whereClause is not None:
            node = Filter(select_stmt.whereClause, node)

        # GROUP BY / Scalar aggregate
        agg_targets = []
        for target in select_stmt.targetList:
            agg_finder = AggFinder()
            agg_finder(target)
            agg_targets.append(agg_finder.has_agg)

        if select_stmt.groupClause is not None or any(agg_targets):
            node = Agg(
                select_stmt.targetList, select_stmt.groupClause, agg_targets, node
            )
        else:
            # SELECT
            assert (
                select_stmt.targetList is not None and len(select_stmt.targetList) > 0
            )
            node = Project(select_stmt.targetList, node)

        # ORDER BY
        if select_stmt.sortClause is not None:
            node = OrderBy(select_stmt.sortClause, node)

        # LIMIT
        if select_stmt.limitCount is not None:
            node = Limit(select_stmt, node)

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

    query = """SELECT maxsolditem
  FROM (SELECT ss_item_sk AS maxsolditem
          FROM (SELECT ss_item_sk
                     , SUM(cnt) AS totalcnt
                  FROM ((SELECT ss_item_sk
                              , COUNT(*) AS cnt
                           FROM store_sales_history
                          GROUP BY ss_item_sk

                          UNION ALL

                         SELECT cs_item_sk
                              , COUNT(*) AS cnt
                           FROM catalog_sales_history
                          GROUP BY cs_item_sk)

                   UNION ALL

                  SELECT ws_item_sk
                       , COUNT(*) AS cnt
                    FROM web_sales_history
                   GROUP BY ws_item_sk) AS t1
                 GROUP BY ss_item_sk) AS t2
         ORDER BY totalcnt DESC
         LIMIT 25000) AS t3
 WHERE getmanufact_complex(maxsolditem) = 'oughtn st';"""

    query = """SELECT ss_item_sk AS maxsolditem
  FROM (SELECT ss_item_sk
             , SUM(cnt) AS totalcnt
          FROM ((SELECT ss_item_sk
                      , COUNT(*) AS cnt
                   FROM store_sales_history
                  GROUP BY ss_item_sk

                  UNION ALL

                 SELECT cs_item_sk
                      , COUNT(*) AS cnt
                   FROM catalog_sales_history
                  GROUP BY cs_item_sk)

           UNION ALL

          SELECT ws_item_sk
               , COUNT(*) AS cnt
            FROM web_sales_history
           GROUP BY ws_item_sk) AS t1
         GROUP BY ss_item_sk) AS t2
 ORDER BY totalcnt DESC
 LIMIT 25000;"""

    query = """SELECT c_customer_sk
     , maxpurchasechannel(c_customer_sk
    , (SELECT MIN(d_date_sk)
         FROM date_dim
        WHERE d_year = 2000)
    , (SELECT MAX(d_date_sk)
         FROM date_dim
        WHERE d_year = 2020)) AS channel
  FROM customer;"""

    print(parse_sql(query))
    print(IndentedStream()(parse_sql(query)[0].stmt))
    plan = Planner.plan_query(query)

    deparsed_ast = plan.deparse()
    print(deparsed_ast)
    print(IndentedStream()(deparsed_ast))
