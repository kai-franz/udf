SELECT month
     , highdeficiencyamount(month)
  FROM GENERATE_SERIES(1, 12) AS _ (month);