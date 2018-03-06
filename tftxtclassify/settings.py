"""project settings."""
import os
# path to main package
PKG_PATH = os.path.dirname(os.path.abspath(__file__))
# project path (one up from main package)
PROJECT_PATH = os.path.abspath(os.path.join(PKG_PATH, '..'))
# path to where data files are located
DATA_PATH = os.path.join(PROJECT_PATH, 'data')
# path to where output files are located (e.g. results of training/experiments)
OUTPUT_PATH = os.path.join(PROJECT_PATH, 'output')
# path to tests
TEST_PATH = os.path.join(os.path.dirname(PKG_PATH), 'tests')

if __name__ == '__main__':
    print(PROJECT_PATH)
