"""project settings."""
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
TEST_DIR = os.path.join(os.path.dirname(BASE_DIR), 'test')

if __name__ == '__main__':
    print(PROJECT_DIR)
