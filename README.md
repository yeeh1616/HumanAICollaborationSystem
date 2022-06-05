1. Activate virtualenv ($source path_to_virtualenv/bin/activate)
2. Go to your project root directory
3. Get all the packages along with dependencies in requirements.txt
    pip3 freeze > requirements.txt
4. You don't have to worry about anything else apart from making sure next person installs the requirements recursively by following command
    pip3 install -r requirements.txt