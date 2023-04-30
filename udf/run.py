# Given a directory containing SQL queries and another directory containing the corresponding UDFs,
# this script will rewrite the UDFs to use batched execution and rewrite the queries to use the new UDFs.
# The rewritten queries and UDFs will be written to the output directory.
#
# Usage: python run.py <query_input_dir> <udf_input_dir> <output_dir>

from rewriter import Rewriter
import sys
import time

query_input_dir = sys.argv[1]
udf_input_dir = sys.argv[2]
output_dir = sys.argv[3]

udfs = [1, 5, 6, 7, 12, 13]
files = [f"sudf_{udf}.sql" for udf in udfs]

rewriting_times = []


def rewrite_query(query_file_name, udf_file_name, output_file_name):
    print("Rewriting", query_file_name)
    with open(query_file_name, "r") as query_file:
        query = query_file.read()
    with open(udf_file_name, "r") as udf_file:
        udf = udf_file.read()
    before_time = time.time()
    for i in range(100):
        r = Rewriter(query, udf, remove_laterals=True)
    after_time = time.time()
    rewriting_times.append(after_time - before_time)
    with open(output_file_name, "w") as out_file:
        out_file.write(r.new_udf())
        out_file.write("\n\n\n")
        out_file.write(r.new_query())


for file in files:
    rewrite_query(
        f"{query_input_dir}/{file}", f"{udf_input_dir}/{file}", f"{output_dir}/{file}"
    )

print("Rewriting times:", rewriting_times)
