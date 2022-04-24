import pandas as pd
# split="train"
split="valid"
sent = pd.read_csv("data/y_yelp_"+split+".csv")
singplur = pd.read_csv("data/SingularPlural/y_yelp_"+split+".csv")
sent_singplur = pd.concat([sent,singplur], axis=1)
sent_singplur.to_csv("data/y_yelp_"+split+"_sent_singplur.csv", index=False)
