"""
File which read and print structure json file
For launch: python3 print_structure.py your_file.json
"""

import json
import sys


def print_example(data, indent=0):
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


def print_structure(data, indent=0):
    for key, value in data.items():
        print('  ' * indent + str(key))
        if isinstance(value, dict):
            print_structure(value, indent + 1)
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            print_structure(value[0], indent + 1)


def main(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    print("\nStructure JSON:")
    print("```json")
    print_structure(data)
    print("```")

    print("```json")
    print("\nExample JSON:")
    print("```")

    print_example(data)


if __name__ == "__main__":
    main(sys.argv[1])
