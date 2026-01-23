# Team-Hill-And-Friend
The repository for our team for Data Science Algorithms 2!


## Using the environment.yml
This is how we will stay on the same page with dependencies and library versions.
There is an environment.yml file in our repo now that we can all download and update as we work on this project.
You can also setup your conda environment (hillandfriend) for the jupyter notebooks.

### How to import
- Ensure you have miniconda or anaconda installed on your computer

- Pull the github repo

- Run:
```
conda env create -f environment.yml
conda activate hillandfriend
```
This will keep us all up to date and ensure we can work in the same environments for reproducible results

### How to update
- If you `conda install` something to use, be sure to update the environment.yml file

- Run:
```
conda env export --no-builds > environment.yml
```

- Ensure file was updated properly