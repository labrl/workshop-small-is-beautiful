# Workshop on Machine Learning to Analyze SCED Data
 
Please follow the following instructions **prior to workshop**.

## Step 1 – Installing a Python Distribution 

Download and install a Python distribution. We strongly recommend the free version (individual edition) of Anaconda available at [www.anaconda.com](https://www.anaconda.com). The steps below assume that you are using Anaconda. 

## Step 2 - Downloading the Functions File 

You must download the "functions.py" file from the Code folder in Github to your computer. Preferably, put all the file in a new folder and note the location of the folder as you will need it to set your working directory. 

## Step 3 – Creating a New Virtual Environment

You need to create an environment that will have all the packages. To create the environment, write the following code Terminal (Apple or Linux) or Anaconda Prompt (Windows): 

```
conda create -n workshop python=3.9
conda activate workshop
```
From now on, you should write the following code when you open Terminal or Anaconda Prompt to ensure that you are working in the correct environment: 
```
conda activate workshop
```
The last line of your Anaconda Prompt or Terminal screen should begin with "workshop". If it begins with "base", you have not activated your environment correctly.

## Step 4 – Installing Packages

To install the necessary packages. Run the following code sequentially. Whenever you are prompted to Proceed, press on “y” followed by the enter key: 
```
conda install scikit-learn
conda install pandas
conda install spyder
conda install matplotlib
```

## Step 5 – Opening Integrated Development Environment

To write and run your Python code, we suggest that you the Spyder integrated environment. To open spyder, write the following code

```
spyder
```
The following screen should appear: 

![Spyder Screenshot](https://raw.githubusercontent.com/labrl/workshop-machine-learning/master/Assets/spyder.png)


You should first set the working the directory. The working directory is the folder in which you have downloaded or saved your data (.csv) files. To do so, you may click on the opened folder icon in the upper right corner and select the appropriate folder.

Enter all the Python code in the left panel of the screen. To run your code, highlight it and click on the “Run selection or current line” or the ![Run Selection Button](https://raw.githubusercontent.com/labrl/workshop-machine-learning/master/Assets/button.png) button in the toolbar. When you run code, any warnings, errors, or printed information will appear in the console (lower right panel). You may also explore the variables in the Variable Explorer in the upper right panel. 

Note that all code entered in Python is case sensitive. 


These instructions were adapted with permission from “Tutorial: Applying machine learning in behavioral research” by S. Turgeon and M. J. Lanovaz, 2020, Perspectives on Behavior Science, [https://doi.org/10.1007/s40614-020-00270-y](https://doi.org/10.1007/s40614-020-00270-y). © Association for Behavior analysis International. 
 

 
