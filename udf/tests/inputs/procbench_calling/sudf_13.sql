SELECT c_customer_sk
     , maxpurchasechannel(c_customer_sk
    , (SELECT MIN(d_date_sk)
         FROM date_dim
        WHERE d_year = 2000)
    , (SELECT MAX(d_date_sk)
         FROM date_dim
        WHERE d_year = 2020)) AS channel
  FROM customer;