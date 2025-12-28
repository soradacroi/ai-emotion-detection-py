import csv
import re

FILE_PATH = 'tweet_emotions.csv'
OUTPUT_FILE_PATH = 'tweet_emotions_cleaned.csv'
CONTENT_COLUMN_INDEX = 1

filter_pattern = re.compile(r'(?:https?://\S+|www\.\S+|@\w+)')

def check_for_filter_content(text):
    if filter_pattern.search(text):
        return True
    return False

output_data = []


with open(FILE_PATH, mode='r') as infile:
    reader = csv.reader(infile)
    header = next(reader)
    output_data.append(header)

    for i, row in enumerate(reader):
        if len(row) > CONTENT_COLUMN_INDEX:
            if row[0] in ["anger", "empty", "hate", "boredom"]:
                row[0] = "negative"
            row = [row[0], re.sub(r"[^a-z ]", "", row[1].lower())]
            original_content = row[CONTENT_COLUMN_INDEX]
            if check_for_filter_content(original_content):
                continue
            
            output_data.append(row)
            


with open(OUTPUT_FILE_PATH, mode='w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(output_data)

print(f"Total rows: {len(output_data) - 1}")
