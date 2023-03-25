SELECT maxsolditem
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
 WHERE getmanufact_complex(maxsolditem) = 'oughtn st';