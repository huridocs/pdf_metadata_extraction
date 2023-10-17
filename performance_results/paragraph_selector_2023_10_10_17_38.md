                          ╷                     ╷        ╷       ╷      ╷       
                          │                     │  Train │  Test │      │       
  Dataset                 │ Method              │   size │  size │ Time │ Acc.  
 ═════════════════════════╪═════════════════════╪════════╪═══════╪══════╪══════ 
  title                   │ base_frequent_words │     15 │    11 │   2s │  83%  
  decides                 │ base_frequent_words │     82 │    55 │  50s │  84%  
  signatories             │ base_frequent_words │     82 │    55 │   5s │  97%  
  first paragraph having  │ base_frequent_words │     82 │    55 │  15s │  96%  
  seen                    │                     │        │       │      │       
  secretary               │ base_frequent_words │     82 │    55 │   4s │  96%  
  president               │ base_frequent_words │     82 │    55 │   4s │  97%  
  date                    │ base_frequent_words │     82 │    55 │   6s │  98%  
  plan many date          │ base_frequent_words │     95 │    64 │   9s │  98%  
  plan many title         │ base_frequent_words │     95 │    64 │  10s │  99%  
  semantic president      │ base_frequent_words │    150 │   100 │ 313s │  76%  
  Average                 │                     │      0 │     0 │   0s │  93%  


  title                   │ next_previous_title │     15 │    11 │   1s │  83%  
  decides                 │ next_previous_title │     82 │    55 │  53s │  81%  
  signatories             │ next_previous_title │     82 │    55 │   5s │  98%  
  first paragraph having  │ next_previous_title │     82 │    55 │  15s │  96%  
  seen                    │                     │        │       │      │       
  secretary               │ next_previous_title │     82 │    55 │   4s │  96%  
  president               │ next_previous_title │     82 │    55 │   4s │  97%  
  date                    │ next_previous_title │     82 │    55 │   6s │  95%  
  plan many date          │ next_previous_title │     95 │    64 │   9s │  99%  
  plan many title         │ next_previous_title │     95 │    64 │  10s │  99%  
  semantic president      │ next_previous_title │    150 │   100 │ 325s │  77%  
  Average                 │                     │      0 │     0 │   0s │  92%   


  title                      │ frequent_6_words │     15 │    11 │   2s │  83%  
  decides                    │ frequent_6_words │     82 │    55 │  56s │  84%  
  signatories                │ frequent_6_words │     82 │    55 │   6s │  98%  
  first paragraph having     │ frequent_6_words │     82 │    55 │  16s │  96%  
  seen                       │                  │        │       │      │       
  secretary                  │ frequent_6_words │     82 │    55 │   5s │  96%  
  president                  │ frequent_6_words │     82 │    55 │   4s │  97%  
  date                       │ frequent_6_words │     82 │    55 │   6s │  97%  
  plan many date             │ frequent_6_words │     95 │    64 │  10s │  98%  
  plan many title            │ frequent_6_words │     95 │    64 │  11s │  99%  
  semantic president         │ frequent_6_words │    150 │   100 │ 332s │  76%  
  Average                    │                  │      0 │     0 │   0s │  92%  

  title                       │ best_features │     15 │    11 │   2s │  83%  
  decides                     │ best_features │     82 │    55 │  51s │  85%  
  signatories                 │ best_features │     82 │    55 │   7s │  98%  
  first paragraph having seen │ best_features │     82 │    55 │  15s │  94%  
  secretary                   │ best_features │     82 │    55 │   7s │  96%  
  president                   │ best_features │     82 │    55 │   5s │  97%  
  date                        │ best_features │     82 │    55 │   7s │  97%  
  plan many date              │ best_features │     95 │    64 │  10s │  99%  
  plan many title             │ best_features │     95 │    64 │  12s │  99%  
  semantic president          │ best_features │    150 │   100 │ 303s │  77%  
  Average                     │               │      0 │     0 │   0s │  92% 

  title                      │ best_features_10 │     15 │    11 │   2s │  83%  
  decides                    │ best_features_10 │     82 │    55 │  49s │  85%  
  signatories                │ best_features_10 │     82 │    55 │   6s │  97%  
  first paragraph having     │ best_features_10 │     82 │    55 │  21s │  96%  
  seen                       │                  │        │       │      │       
  secretary                  │ best_features_10 │     82 │    55 │   5s │  96%  
  president                  │ best_features_10 │     82 │    55 │   5s │  97%  
  date                       │ best_features_10 │     82 │    55 │   7s │  97%  
  plan many date             │ best_features_10 │     95 │    64 │  11s │  98%  
  plan many title            │ best_features_10 │     95 │    64 │  15s │  99%  
  semantic president         │ best_features_10 │    150 │   100 │ 304s │  76%  
  Average                    │                  │      0 │     0 │   0s │  93% 

  title                       │ avoiding_words │     15 │    11 │   2s │  83%  
  decides                     │ avoiding_words │     82 │    55 │ 131s │  86%  
  signatories                 │ avoiding_words │     82 │    55 │   9s │  96%  
  first paragraph having seen │ avoiding_words │     82 │    55 │  21s │  96%  
  secretary                   │ avoiding_words │     82 │    55 │   5s │  96%  
  president                   │ avoiding_words │     82 │    55 │   5s │  97%  
  date                        │ avoiding_words │     82 │    55 │   8s │  97%  
  plan many date              │ avoiding_words │     95 │    64 │  23s │  98%  
  plan many title             │ avoiding_words │     95 │    64 │  13s │  99%  
  semantic president          │ avoiding_words │    150 │   100 │ 575s │  77%  
  Average                     │                │      0 │     0 │   0s │  93%