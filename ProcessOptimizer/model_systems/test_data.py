import csv
import os

def test_data():
    package_directory = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(package_directory, 'data', 'test_data.csv')
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
    print(headers)