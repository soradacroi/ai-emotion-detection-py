import textmood

LABELS = ['sadness', 'happiness', 'neutral']

STOP_WORDS = {"the", "a", "an", "is", "and", "or", "to", "of", "in", "it"}
NEGATIONS = {"not"}

textmood.no_words(STOP_WORDS, NEGATIONS)

def main():
    print(f"training started...")
    
    data = textmood.load_data('tweet_emotions_cleaned.csv', 30000, LABELS, ",")

    train_data, test_data = textmood.partition_samples(data, split_p=0.8)
    print(f"train samples: {len(train_data)}, test samples: {len(test_data)}")
    
    textmood.train(train_data, LABELS)

    test_phrases = {
        "i am not happy": "sadness",
        "i am not sad but happy": "happiness",
        "i am so excited for the test tommorow": "happiness",
        "hello": "neutral"
    }

    for phrase, actual in test_phrases.items():
        prediction = textmood.predict(phrase, LABELS)
        print(f"\"{phrase}\"")
        print(f"Predicted: {prediction}, Actual: {actual}\n")

    correct = 0
    for content, actual_label in test_data:
        prediction = textmood.predict(content, LABELS)
        if prediction == actual_label:
            correct += 1

    accuracy = (correct / len(test_data)) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct} / {len(test_data)}")

    textmood.save_model("emotion_model.pkl")

if __name__ == "__main__":
    main()