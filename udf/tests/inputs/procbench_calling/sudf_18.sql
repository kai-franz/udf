SELECT c_customer_sk
     , preferredchannel_wrtexpenditure(c_customer_sk)
  FROM customer;