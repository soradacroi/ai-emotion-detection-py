# ai-emotion-detection-py


the data which i will be using is a public domain data from data.world, data consist of tweet and emotions


// after clearing the data a little this is the count of emotions (how many items we have for each emotion)

after seeing this data idk... ehhh... it looks too like worry has 36636 items and sentiment only 7... i might remove some items. but later

Emotion counts:  
sadness: 23784   
enthusiasm: 2736  
neutral: 29492   
worry: 36636   
surprise: 7652   
fun: 5840   
hate: 6288   
happiness: 19672   
boredom: 1020   
love: 14036   
relief: 6452   
empty: 3148   
anger: 488   
sentiment: 7  


why there is so much worries remove that, let some happiness count and make some fun(5840) and love(14036) -> happiness(19672) = 39548

so now we will have neutral : 29492 then sadness :  23784 then happiness : 39548 we need one more to make it interesting.

BTW im doing this all because the data for some emotion is way too much and some are way too little which is making my models biased towards some emotions like **worry**, i worte "i am happy" and it gave **worry** like what the hell... also i was not doing any mistakes in coding part as i just ~copied~ took some inspiration from someone elese code, ok no jokes i just took some idea on to how to do it using
[geeksforgeek](https://www.geeksforgeeks.org/nlp/word-embeddings-in-nlp/) idk if its good or not... but it yeah it had some bias predictions.

so yeah the another emotion i want to select for is anger but man its too small data, maybe combining (anger, empty, hate, boredom) will help as it will make a dataset of 10k elements **i called this dataset negative**, and that will be enough for us to do. Using 10k elements from all of the dataset.

That can also be good as we now only need 10k so we can use **love** as other category too, and **worry** all 10k.

I'm not removing the first "why there is so much worries..." tho as its like i liked that line.

ok so the numbers of counts for emotions are like this now

sadness: 23784            ->  
enthusiasm: 2736  
neutral: 29492            ->  
worry: 36636              ->  
surprise: 7652  
fun: 5840  
negative: 10944           ->  
happiness: 19672          ->  
love: 14028               ->      
relief: 6452  
sentiment: 7  

we can use sadness, neutral, worry, negative, love and happiness now, as these all has more then 10k elements
and maybe using a data which has same numbers of elements for each category will help make a better model

ALSO i will be trying to come up with something else instead of using TF-IDF. 

I HOPE I CAN DO THAT BY MYSELF 