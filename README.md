# OpenBATS

OpenBATS is a Python tool for optimizing building energy usage and cost (plus other additional optimizations) utilizing AI models built with data collected for a given building requiring no traditional building models.

## Requirements

The current code has been developed and tested using Python 3.7.12 on a Windows 10 machine. To be able to simplify the dependencies and replicate the code in the project, a specific environment was created using Anaconda3 (Windows 10 Anaconda3 version 2.3.2), and using a conda version of 22.9.0. An environment file **environment.yml** has been included as part of the project files to replicate the environment in Anaconda3. 

The project execution is based on pytorch and uses Cuda library for computing. First create an environment in Anaconda3 using the provided **environment.yml** file. Then open up a conda command prompt, use the code below to create a python and execute the following code to verify that CUDA is available on the windows 10 machine before attempting to execute the code in this project. 

```python
import torch

if torch.cuda.is_available():
	print("CUDA is available")
else:
	print("CUDA is not Available")
```


## Installation and Usage

1. Download this repository
2. Install Anaconda3 
3. Create an new Anaconda environment using the **environment.yml** file. 
4. Open an Anaconda command prompt (at installation you could opt to include the paths to the Anaconda3 executables and scripts)
5. Navigate to the directory where you downloaded the repository files.
6. Run the python script file openbats_v10, and follow the instruction in the file **Software_User_Manual_ver_10.pdf**

```python
python openbats_v10.py
```

## License

[MIT](https://choosealicense.com/licenses/mit/)


##Acknowledgment
This work was funded by California Energy Commission’s (CEC) Energy Research and Development Division, and our sincere thanks to CEC for their engagement and support of this work.