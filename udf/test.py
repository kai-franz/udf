from query_rewriter import *
from udf_rewriter import *

q_no_filter = """select c_customer_sk, increaseInWebSpending(c_customer_sk, bill_customer_sk)
from customer
where c_customer_sk in
	(select ws_bill_customer_sk
	from web_sales_history, date_dim
	where d_date_sk = ws_sold_date_sk
		and d_year = 2000

	INTERSECT

	select ws_bill_customer_sk
	from web_sales_history, date_dim
	where d_date_sk = ws_sold_date_sk
		and d_year = 2001
	)"""

f = """
CREATE OR REPLACE FUNCTION increaseInWebSpending(cust_sk INT)
    RETURNS DECIMAL
    LANGUAGE plpgsql
AS
$$
DECLARE
    spending1 DECIMAL;
    spending2 DECIMAL;
    increase  DECIMAL;
BEGIN
    spending1 := 0;
    spending2 := 0;
    increase := 0;

    SELECT SUM(ws_net_paid_inc_ship_tax)
      INTO spending1
      FROM web_sales_history,
           date_dim
     WHERE d_date_sk = ws_sold_date_sk
       AND d_year = 2001
       AND ws_bill_customer_sk = cust_sk;

    SELECT SUM(ws_net_paid_inc_ship_tax)
      INTO spending2
      FROM web_sales_history,
           date_dim
     WHERE d_date_sk = ws_sold_date_sk
       AND d_year = 2000
       AND ws_bill_customer_sk = cust_sk;

    IF (spending1 < spending2) THEN
        RETURN -1;
    ELSE
        increase := spending1 - spending2;
    END IF;
    RETURN increase;

END;
$$;"""

root_q = parse_sql(q_no_filter)
transformQuery(root_q)
with open("query_out.sql", "w") as query_out:
    query_out.write(IndentedStream()(root_q))
root = parse_plpgsql(f)[0]["PLpgSQL_function"]
rewriter = UdfRewriter(f)
with open("udf_out.sql", "w") as udf_out:
    udf_out.write(rewriter.output())
