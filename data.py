import csv
from collections import Counter

CSV_NAME = "tweet_emotions_cleaned.csv"

counts = Counter()

with open(CSV_NAME, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        counts[row["sentiment"]] += 1

print("Emotion counts:")
for emotion, count in counts.items():
    print(f"{emotion}: {count}")
