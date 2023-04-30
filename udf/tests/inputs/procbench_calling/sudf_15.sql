SELECT s_store_sk
  FROM store
 WHERE incomebandofmaxbuycustomer(s_store_sk) = 'lowerMiddle'
 ORDER BY s_store_sk;