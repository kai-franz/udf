from query_rewriter import *
from udf_rewriter import *


class Rewriter:
    def __init__(self, query, udf):
        self.query = query
        self.udf = udf
        self.schema = Schema()

        # Do the rewrites
        self.udf_rewriter = UdfRewriter(udf, self.schema)
        self.root_q = parse_sql(query)
        transformQuery(self.root_q, self.udf_rewriter.original_func_name)

    def new_query(self):
        return IndentedStream()(self.root_q)

    def new_udf(self):
        return self.udf_rewriter.output()
