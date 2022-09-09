# splade-colBERT
Capstone

# Set up

**Cloning the Repo**

To clone the repo properly run:
`git clone [repo URL from github] --recursive`

**Using Conda + Poetry:**

1. `[Optional]` Install conda, an environment and package controller. There are 2 distributions that will install conda to your computer: Anaconda and Miniconda.

   - 1.1 [anaconda](https://www.anaconda.com/download/) - includes 1000+ data
     science libraries
   - 1.2 [miniconda](https://conda.io/miniconda.html) - barebones installation \
     Note: Learn more about [Conda](https://conda.io/docs/user-guide/overview.html)

2. Set up your environment with the proper python version(3.9) and dependencies.

   2.1 - Create a new environment from the `torch_environment.yml` file:
   `conda env create -f torch_environment.yml`\

   - The default environment name will be `capstone`.

     2.2 - Activate the newly created environment:

   - For Linux/Macos: `source activate [environment_name]`
   - For Windows: `activate [environment_name]`

     2.3 - Install dependencies on an already created environment:
     `conda env update --file environment.yml`

3. `[Optional]` If you newly created environment isn't active, do `Step 2.2`. \
   Or use `conda run [-n [environment_name]]`.

4. Install all package dependencies:
   `poetry install`
