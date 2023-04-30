SELECT ws_item_sk
  FROM (SELECT ws_item_sk
             , COUNT(*) AS cnt
          FROM web_sales
         GROUP BY ws_item_sk
         ORDER BY cnt
         LIMIT 25000) AS t1
 WHERE getmanufact_simple(ws_item_sk) = 'oughtn st';