# SALARY PREDICTION V 1.0

This is a simple salary prediction model using linear regression in this case ExtraTreesRegressor.
The model is trained on a dataset containing various features that may influence salary, such as years of experience, education level, and job title.

## Step 0: Clone the repository
First at all if you use windows, you need to install git bash, a terminal that allows you to run bash commands on Windows.
You can download it from the official website: [Git for Windows](https://gitforwindows.org/).

To get started with the project, you need to clone the repository to your local machine.
Please locate the directory where you want to clone the repository and run the following command:
You can do this by running the following command in your terminal or command prompt:
1. Clone the repository:
   ```bash
   git clone https://github.com/leonardoheis/fau_predic_salario.git
    ```
2. Navigate to the project directory where the repository was cloned:

## Step 1: Install the right version of Python and other dependencies

### Windows Users: (Optional)
Make sure in App and Programs you don't have any version of Python installed. If you have, uninstall it.
Make sure in the Aliases for Applications you have Python unchecked.
For install a new version of Python, you can download it from the official website: [Python Downloads](https://www.python.org/downloads/) or
in this case use chocolatey, a package manager for Windows.
To install chocolatey, open a command prompt as administrator and run:
```bash
@"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))" && SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"
```
After installing chocolatey, you can install Python by running:
```bash
choco install python --version=3.10.11 -y --force
```
This version is not what we used to develop this project, but is the unique version that could nbe installed with chocolatey in windows.
The right version should be 3.10.17, but it is not available in chocolatey for windows.

### MacOS and Linux Users: (Optional)
Be careful to not uninstall the system Python, as it is required for many system operations. It should be safe to remove any user-installed Python versions.
For MacOS, you can use Homebrew to manage Python uninstallation.
If don't have Homebrew installed, you can install it by running:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
Wih Homebrew installed, you can uninstall Python by running:
```bash
brew uninstall python@3.10
```
Indicates the right version of Python to uninstall, if you have more than one version installed.

On the other hand, now you can install the right version of Python, 3.10.17, by running:
```bash
brew install python@3.10
```
## Step 1: Install Pipenv

### Windows Users:
To install Pipenv, within the git bash terminal and posicionate where the project was cloned, execute the following command:
```bash
pip install pipenv
```
you should have installed pip in your system, you can check it by running:
```bash
pip --version
```
### MacOS and Linux Users:


## Step 2: Create a virtual environment with Pipenv
First of all you need to install the right version of Python, 3.10.11 for windows and 3.10.17 for MacOS or Linux, 
within the environment where you want to run the project.
Then, within the git bash terminal and posicionate where the project was cloned, execute the following command:
```bash
pipenv --python /c/Python310/python.exe
```
The ```/c/Python310/python.exe``` is the path where Python was installed by chocolatey.

With this you can create a virtual environment with the right version of Python.
For both cases the command is the same, but you need to change the path to the Python executable according to your system.
This should e lik ethis because you could have more than one python installed, but if it is the only one, you can just run the command without the path explicitelly.

## Step 3: Install the dependencies
Now that you have created the virtual environment, you need to install the dependencies required for the project.
For this cases positioned in the project directory, run the following command:
```bash
pipenv install
```
This will install all the dependencies listed in the `Pipfile` and `Pipfile.lock` files.

## Step 4: Run the project
Consider this project it is separated in front and back end, so you need to run both parts.
before initiating the project, you need to run the following command to activate the virtual environment:
```bash
pipenv shell
```
### Backend
Open a terminal and run the following command:
```bash
pipenv --verbose run start-api
```
This will start the backend server, which will listen for requests on the specified port (default is 8000).

### Frontend
Open another terminal and run the following command:
```bash
pipenv --verbose run start-ui
```
This will start the frontend server, which will serve the user interface for the salary prediction application.

## Step 5: Access the application
Once both the backend and frontend servers are running, you can access the application by opening your web browser and navigating to:
```
Frontend: http://localhost:8501
Backend: http://127.0.0.1:8000/docs  in this case you can see the API documentation and test the endpoints that Swagger UI provides.
```





