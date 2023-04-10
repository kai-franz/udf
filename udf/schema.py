from pglast import ast, parse_sql
from pglast.enums import ConstrType


class Table:
    def __init__(self, name, cols: dict, primary_key: str = None):
        self.name = name
        self.cols = cols
        self.primary_key = primary_key


class Schema:
    """
    Represents the schema of the database.
    """

    def __init__(self, schema_file):
        self.tables = {}
        with open(schema_file, "r") as f:
            schema_ast = parse_sql(f.read())
        for raw_stmt in schema_ast:
            stmt = raw_stmt.stmt
            if not isinstance(stmt, ast.CreateStmt):
                continue
            self.add_table_from_ast(stmt)

    def add_table(self, name, cols: dict, primary_key: str = None):
        self.tables[name] = Table(name, cols, primary_key)

    def add_table_from_ast(self, table_ast: ast.CreateStmt):
        name = table_ast.relation.relname
        cols = {}
        primary_key = None
        for col in table_ast.tableElts:
            if isinstance(col, ast.ColumnDef):
                cols[col.colname] = col
                if col.constraints:
                    for constraint in col.constraints:
                        assert isinstance(constraint, ast.Constraint)
                        if constraint.contype == ConstrType.CONSTR_PRIMARY:
                            assert primary_key is None
                            primary_key = col.colname
        self.add_table(name, cols, primary_key=primary_key)

    def get_columns_for_table(self, table):
        return set(self.tables[table].cols.keys())

    def get_columns(self, tables):
        columns = set()
        for table in tables:
            columns = columns.union(self.get_columns_for_table(table))
        return columns

    def get_primary_key(self, table):
        return self.tables[table].primary_key


class ProcBenchSchema(Schema):
    def __init__(self):
        super().__init__("./tests/inputs/procbench_schema.sql")
