import os
import sys

try:
    import tslearn
    print("tslearn is available!")
    print(f"Python executable: {sys.executable}")
    print(f"Environment PATH: {os.environ['PATH']}")
except ModuleNotFoundError as e:
    print(f"Module not found: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
