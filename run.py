import sys
import os
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    script = os.path.join(BASE_DIR, "Source code", "Final_detector.py")
    subprocess.run([sys.executable, script])

if __name__ == '__main__':
    main()
