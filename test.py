import json

# Open the JSON file and load its contents into a dictionary
with open('intents.json', 'r') as f:
    data = json.load(f)

# Print the dictionary in a formatted way
print(json.dumps(data, indent=4))

