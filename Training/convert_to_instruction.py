import json

INPUT_FILE = "training_data.jsonl"
OUTPUT_FILE = "instruction_data.jsonl"


with open(INPUT_FILE) as f:
    lines = f.readlines()


out = open(OUTPUT_FILE, "w")


for line in lines:
    item = json.loads(line)

    new_item = {
        "instruction": item["question"],
        "input": item["context"],
        "output": item["answer"]
    }

    out.write(json.dumps(new_item) + "\n")


out.close()

print("Converted to instruction format.")
