import math, random, pickle

STOP_WORDS = set()
NEGATIONS = set()

def no_words(stop_w: set, negation_w : set):
    global STOP_WORDS, NEGATIONS
    STOP_WORDS = stop_w
    NEGATIONS = negation_w

def get_features(text: str):
    raw_words = ""
    for char in text.lower():
        if 'a' <= char <= 'z':
            raw_words += char
        else:
            raw_words += " "
    
    tokens = raw_words.split()
    
    processed_tokens = []
    negate = False
    for t in tokens:
        if t in STOP_WORDS: continue
        
        if t in NEGATIONS:
            negate = True
            continue
        
        if negate:
            processed_tokens.append("NOT_" + t)
            negate = False 
        else:
            processed_tokens.append(t)
            
    # add grams or what idk what its called 
    grams = []
    for i in range(len(processed_tokens) - 1):
        grams.append(processed_tokens[i] + processed_tokens[1+i]) # this is just bi gram
    for i in range(len(processed_tokens) - 2):
        grams.append(processed_tokens[i] + processed_tokens[1+i] + processed_tokens[2+i]) # this is tri gram
    # why didnt i added more?? beause it was not improving that much and it takes come compute power u know
    return processed_tokens + grams

# i made the code for well 2 emotions now... (happy and sad) i was thinking to make this like a libary is that correct word?
# so like i want to be able to train any thing using this file... well idk how i will do this lets just do it and fugure it later
# btw i love ADOOOOOOOOOOOOOOOOOOOOOOO https://open.spotify.com/intl-ja/track/3pncyVfDzI4m9oErJKLKIo?si=6a729ed0254e48dd

# this is what do the loading data part so... like it takes path of a csv file, the separator and lebels 
# file can be csv, txt or anything just that it should have two elements like : 
# one element, other element ("," can be any separator)
# then it returns the samples in formate of tuples inside of a list, where the elemets of list is like :
# (content, lebel)
def load_data(data_file_path: str, amount: int, labels: list, separator: str):
    all_samples = []
    labels_set = set(labels)
    
    # Track how many of each label we have collected
    counts = {label: 0 for label in labels}
    
    with open(data_file_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.split(separator, 1)
            if len(parts) < 2:
                continue
                
            label = parts[0].strip()
            content = parts[1].strip()
            
            # Check if this label is one we want AND if we still need more of it
            if label in labels_set and counts[label] < amount:
                all_samples.append((content, label))
                counts[label] += 1
            
            # Optimization: if all labels have reached the 'amount', we can stop reading
            if all(c >= amount for c in counts.values()):
                break
                
    return all_samples


def partition_samples(samples, split_p = 0.8, seed = 42):
    random.seed(seed) 
    random.shuffle(samples)
    split_idx = int(len(samples) * split_p)
    train_data = samples[:split_idx]
    test_data = samples[split_idx:]
    return train_data, test_data

counts = {}
total_feat_count = {}
docs_per_class = {}
vocab = set()
priors = {}
total_docs = -1

def train(train_data, labels, say=True):
    global vocab, priors, counts, total_feat_counts, docs_per_class

    for label in labels:
        counts[label] = {}
        total_feat_count[label] = 0
        docs_per_class[label] = 0

    for text, label in train_data:
        docs_per_class[label] += 1
        features = get_features(text)
        for feat in features:
            counts[label][feat] = counts[label].get(feat, 0) + 1
            total_feat_count[label] += 1
            vocab.add(feat)

    # calculate priors
    total_docs = sum(docs_per_class.values())
    for label in labels:
        if total_docs > 0 and docs_per_class[label] > 0:
            priors[label] = math.log(docs_per_class[label] / total_docs)
        else:
            priors[label] = -1e9 # smoll num for empty classes
            
    if say:
        print(f"training done vocab size: {len(vocab)}\n")


def save_model(name = "model.pkl"):
    data = {
        "counts": counts,
        "total_feat_count": total_feat_count,
        "docs_per_class": docs_per_class,
        "priors": priors,
        "vocab": vocab,
        "labels": list(priors.keys())
    }
    with open(name, "wb") as file:
        pickle.dump(data, file)
    print(f"Model saved successfully as {name}")
    
def save_model(name="emotion_model.pkl"):
    # We use a dictionary to keep things organized
    # We pull labels from the priors keys so you don't need a global LABELS list
    model_data = {
        "counts": counts,
        "total_feat_count": total_feat_count,
        "docs_per_class": docs_per_class,
        "priors": priors,
        "vocab": vocab,
        "labels": list(priors.keys())
    }
    with open(name, "wb") as file:
        pickle.dump(model_data, file)
    print(f"Model saved to {name}")

def load_model(name="emotion_model.pkl"):
    global counts, total_feat_count, docs_per_class, priors, vocab
    
    with open(name, "rb") as file:
        data = pickle.load(file)
        
    counts = data["counts"]
    total_feat_count = data["total_feat_count"]
    docs_per_class = data["docs_per_class"]
    priors = data["priors"]
    vocab = data["vocab"]
    
    print(f"Model loaded successfully! Vocab size: {len(vocab)}")
    return data["labels"]

def predict(text, labels, smooth=0.1):
    features = get_features(text)
    vocab_size = len(vocab)
    
    scores = {}

    for label in labels:
        # log of the prior: log(P(Class))
        scores[label] = priors[label]
        
        total_feat_in_class = total_feat_count[label]
        
        for feat in features:
            feat_count_in_class = counts[label].get(feat, 0)
            
            # P(Feature | Class) with "Laplace Smoothing " <- idk this thing
            # formula: (count + smooth) / (total_count_in_class + vocab_size * smooth)
            p_feat_given_class = (feat_count_in_class + smooth) / (total_feat_in_class + (vocab_size * smooth))
            scores[label] += math.log(p_feat_given_class)

    return max(scores, key=scores.get)

