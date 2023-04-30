SELECT c_customer_sk
     , preferredchannel_wrtcount(c_customer_sk)
  FROM customer;