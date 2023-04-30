import unittest

from pglast.stream import IndentedStream

from udf.planner import Planner
from pglast import parser
import os
import natsort

simple_queries = [
    "select * from customer inner join web_sales on customer.c_customer_sk = web_sales.ws_sold_date_sk;",
    "select * from (select * from customer inner join web_sales on customer.c_customer_sk = web_sales.ws_sold_date_sk) as t1 where c_customer_sk > 1000 and c_customer_sk < 10000;",
    """SELECT * FROM (SELECT ss_item_sk
                              , COUNT(*) AS cnt
                           FROM store_sales_history
                          GROUP BY ss_item_sk

                          UNION ALL

                         SELECT cs_item_sk
                              , COUNT(*) AS cnt
                           FROM catalog_sales_history
                          GROUP BY cs_item_sk) AS t1;""",
]

# Get the queries in the files in the folder udf/tests/inputs/procbench_calling
# and add them to the list of queries to test.
procbench_calling_dir = os.path.join(
    os.path.dirname(__file__), "inputs", "procbench_calling"
)
query_file_names = natsort.natsorted(
    [file for file in os.listdir(procbench_calling_dir) if file.endswith(".sql")]
)
procbench_queries = []
for file_name in query_file_names:
    with open(os.path.join(procbench_calling_dir, file_name), "r") as f:
        procbench_queries.append(f.read())


class TestPlanner(unittest.TestCase):
    def test_simple(self):
        for query in simple_queries:
            print("Testing simple query: " + query)
            plan = Planner.plan_query(query)
            expected_fingerprint = parser.fingerprint(query)
            planner_fingerprint = parser.fingerprint(IndentedStream()(plan.deparse()))
            self.assertEqual(expected_fingerprint, planner_fingerprint)

    def test_procbench_calling(self):
        skip_queries = {}  # {"sudf_1.sql"}
        test_queries = {f"sudf_{i}.sql" for i in [1, 5, 6, 7, 12, 13]}
        for name, query in zip(query_file_names, procbench_queries):
            if name in skip_queries or name not in test_queries:
                continue
            print("Testing query: " + name)
            plan = Planner.plan_query(query)
            expected_fingerprint = parser.fingerprint(query)
            planner_fingerprint = parser.fingerprint(IndentedStream()(plan.deparse()))
            self.assertEqual(expected_fingerprint, planner_fingerprint)


if __name__ == "__main__":
    unittest.main()
