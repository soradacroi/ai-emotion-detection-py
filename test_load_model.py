import textmood

# 1. Setup the same rules used during training
STOP_WORDS = {"the", "a", "an", "is", "and", "or", "to", "of", "in", "it"}
NEGATIONS = {"not"}
textmood.no_words(STOP_WORDS, NEGATIONS)

loaded_labels = textmood.load_model("emotion_model.pkl")

def ask_mood():

    
    while True:
        user_input = input("\ninput> ")
        
        if user_input.lower() == 'quit':
            break
            
        prediction = textmood.predict(user_input, loaded_labels)
        
        print(prediction.upper())

if __name__ == "__main__":
    ask_mood()