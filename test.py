print('cuda' if __import__('torch').cuda.is_available() else 'cpu')
