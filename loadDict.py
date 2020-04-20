import json

def load_dict():
    with open("dict.json","r") as jsonfile:
        pairs = json.load(jsonfile)

        return pairs['pairs']
        # for pair in pairs['pairs']:
        #     print(pair['icon'])
        #     print(pair['model'])


if __name__ == "__main__":
    load_dict()