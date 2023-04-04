from pglast import ast


class Schema:
    """
    Represents the schema of the database.
    """

    def __init__(self):
        self.tables = {}

    def add_table(self, name, cols: dict):
        self.tables[name] = cols

    def add_table_from_ast(self, table_ast: ast.CreateStmt):
        name = table_ast.relation.relname
        cols = {}
        for col in table_ast.tableElts:
            if isinstance(col, ast.ColumnDef):
                cols[col.colname] = col
        self.add_table(name, cols)

    def get_columns(self, tables):
        columns = set()
        for table in tables:
            columns = columns.union(self.tables[table].keys())
        return columns
