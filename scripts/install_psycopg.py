#!/usr/bin/env python
"""
Script to install psycopg packages correctly.
Run this script to ensure the correct psycopg packages are installed.
"""

import sys
import subprocess
import os
import platform

def main():
    print("Installing psycopg packages...")
    
    # Determine if we need quotes based on the platform
    if platform.system() == "Windows":
        psycopg_cmd = ['pip', 'install', 'psycopg[binary]==3.1.13', 'psycopg-pool==3.1.8']
    else:
        psycopg_cmd = ['pip', 'install', 'psycopg[binary]==3.1.13', 'psycopg-pool==3.1.8']
    
    # Install psycopg3
    try:
        print("Installing psycopg3...")
        subprocess.check_call(psycopg_cmd)
        print("Successfully installed psycopg3 packages")
    except subprocess.CalledProcessError as e:
        print(f"Error installing psycopg3: {e}")
        # Try alternate command with quotes
        try:
            print("Trying alternate installation method...")
            # For Windows CMD which might need quotes
            subprocess.check_call(['pip', 'install', '"psycopg[binary]"==3.1.13', 'psycopg-pool==3.1.8'], shell=True)
            print("Successfully installed psycopg3 packages using alternate method")
        except subprocess.CalledProcessError as e2:
            print(f"Error with alternate installation method: {e2}")
            print("Please manually install the packages with:")
            print('pip install "psycopg[binary]"==3.1.13 psycopg-pool==3.1.8')
            
    # Verify installation
    try:
        subprocess.check_call([sys.executable, '-c', 'import psycopg; print("psycopg3 successfully imported")'])
        print("Verification successful: psycopg3 is installed correctly")
    except subprocess.CalledProcessError:
        print("psycopg3 import test failed")
        
    print("\nScript complete. If you still encounter issues, try manually installing:")
    print('pip install "psycopg[binary]"==3.1.13 psycopg-pool==3.1.8')

if __name__ == "__main__":
    main() 