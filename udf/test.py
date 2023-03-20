from query_rewriter import *
from udf_rewriter import *
import sys

query_input_dir = sys.argv[1]
udf_input_dir = sys.argv[2]
output_dir = sys.argv[3]

udfs = [1, 5, 6, 7, 12, 13]
files = [f"sudf_{udf}.sql" for udf in udfs]


def rewrite_query(query_file_name, udf_file_name, output_file_name):
    print("Rewriting", query_file_name)
    with open(query_file_name, "r") as query_file:
        query = query_file.read()
    with open(udf_file_name, "r") as udf_file:
        udf = udf_file.read()
    root_q = parse_sql(query)
    transformQuery(root_q)
    rewriter = UdfRewriter(udf)
    with open(output_file_name, "w") as out_file:
        out_file.write(rewriter.output())
        out_file.write("\n\n\n")
        out_file.write(IndentedStream()(root_q))


for file in files:
    rewrite_query(
        f"{query_input_dir}/{file}", f"{udf_input_dir}/{file}", f"{output_dir}/{file}"
    )
