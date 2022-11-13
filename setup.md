# Install Anaconda

* Download [Anaconda package](https://repo.anaconda.com/archive/Anaconda3-2022.10-MacOSX-x86_64.pkg)
* Click pkg file to install


# Create Python Environment 
	
	conda create --name ai_fundataion python==3.10.6
	conda env list
	conda activiate ai_fundataion
	conda deactiviate

# Install Python Packages
	
	pip install -r requirements.txt

# Run model training 

	python train.py


# Run model testing

	python test.py

# Install Jupyter Notebook
	
	pip install jupyterlab
	jupyter-lab
