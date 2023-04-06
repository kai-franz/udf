from pglast import ast, parse_sql


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

    def add_table(self, name, cols: dict):
        self.tables[name] = cols

    def add_table_from_ast(self, table_ast: ast.CreateStmt):
        name = table_ast.relation.relname
        cols = {}
        for col in table_ast.tableElts:
            if isinstance(col, ast.ColumnDef):
                cols[col.colname] = col
        self.add_table(name, cols)

    def get_columns_for_table(self, table):
        return set(self.tables[table].keys())

    def get_columns(self, tables):
        columns = set()
        for table in tables:
            columns = columns.union(self.get_columns_for_table(table))
        return columns


class ProcBenchSchema(Schema):
    def __init__(self):
        super().__init__("./tests/inputs/procbench_schema.sql")
