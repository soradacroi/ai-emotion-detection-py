import textmood

# define labels
LABELS = ['worry', 'neutral', 'sadness', 'happiness', ]

# setup stop words and negations
STOP_WORDS = {"the", "a", "an", "is", "and", "or", "to", "of", "in", "it"}
NEGATIONS = {"not", "no", "never", "dont", "cant", "isnt", "wasnt"}
textmood.no_words(STOP_WORDS, NEGATIONS)

def main():
    print(f"training")
    
    # load data
    data = textmood.load_data('tweet_emotions_cleaned.csv',30000 , LABELS, ",")

    # partition data (80% train, 20% test)
    train_data, test_data = textmood.partition_samples(data, split_p=0.8)
    print(len(train_data), len(test_data))
    # train the model
    textmood.train(train_data, LABELS)

    # test it out
    test_phrases = {
        "i am so scared of tommorw's exam": "worry",
        "hello": "neutral",
        "today is a very sad day for me": "sadness",
        "i am so excited for the test tommorow": "happiness"
    }

    print("model predictions:\n")
    for phrase, actual in test_phrases.items():
        prediction = textmood.predict(phrase, LABELS)
        print(f"\"{phrase}\"")
        print(f"predicted: {prediction}, actual: {actual}\n")
    correct = 0
    total = len(test_data)

    # evaluation of well acuracy 
    correct = 0
    total = len(test_data)
    print(f"testing on {total} samples\n")
    for content, actual_label in test_data:
        prediction = textmood.predict(content, LABELS)
        if prediction == actual_label:
            correct += 1

    accuracy = (correct / total) * 100
    print(f"Accuracy: {accuracy}%")
    print(f"Correct: {correct}")
    print(f"Total: {total}")

if __name__ == "__main__":
    main()