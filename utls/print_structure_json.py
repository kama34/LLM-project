import json
import sys


def print_example(data, indent=0):
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                print('  ' * indent + str(key) + ': {')
                print_example(value, indent + 1)
                print('  ' * indent + '}')
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                print('  ' * indent + str(key) + ': [')
                print_example(value[0], indent + 1)
                print('  ' * indent + ']')
            else:
                print('  ' * indent + str(key) + ': ' + str(value))
    elif isinstance(data, list):
        print_example(data[0], indent)


def print_structure(data, indent=0):
    if isinstance(data, dict):
        for key, value in data.items():
            print('  ' * indent + str(key))
            if isinstance(value, dict):
                print_structure(value, indent + 1)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                print_structure(value[0], indent + 1)
    elif isinstance(data, list):
        print_structure(data[0], indent)


def main(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    print("\n**Structure JSON:**")
    print("```json")
    print_structure(data)
    print("```")

    print("\n**Example JSON:**")
    print("```json")
    print_example(data)
    print("```")


if __name__ == "__main__":
    main(sys.argv[1])
