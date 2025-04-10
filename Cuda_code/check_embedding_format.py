import numpy as np

def check_embedding_format(file_path):
    with open(file_path, 'rb') as f:
        header = f.read(8)
        print(f"File header (first 8 bytes): {header}")
        
        try:
            # Try loading with numpy
            arr = np.load(file_path)
            print(f"Successfully loaded as numpy array with shape: {arr.shape}")
            return True
        except:
            print("Not a numpy array format")
            return False

if __name__ == "__main__":
    import sys
    check_embedding_format(sys.argv[1])
