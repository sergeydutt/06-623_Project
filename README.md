# Group Project

Welcome to our group project repository! This repository contains all files needed for the semester project.

## Notebooks

Use the table below to open and run the project notebooks. If you are using **Google Colab**, simply click the **Open in Colab** badge next to the notebook you want to work on.

| Notebook | Description | Launch in Cloud |
| :--- | :--- | :---: |
| **`Project_PoC.ipynb`** | Proof of Concept | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sergeydutt/06-623_Project/blob/main/Project_PoC.ipynb) |

## How to Keep Main Updated

### If using Google Colab

1. Open: Click the **Open in Colab badge** above (or go to File > Open notebook > GitHub tab and select sergeydutt/06-623_Project)

2. Work: Make your edits and run your code.

3. Save: When finished, go to File > Save a copy in GitHub, select the sergeydutt/06-623_Porject repository, choose the main branch, enter a commit message, and click **OK**. 

### If using a local computer (Terminal / Linux)

#### First-time setup (Initialize Git)
Run these commands once inside your project folder to set up the repository:
```bash
git init
git branch -M main
git remote add origin [https://github.com/sergeydutt/06-623_Project.git](https://github.com/sergeydutt/06-623_Project.git)
git pull origin main

#### Regular Instructions
Follow these commands everytime you work on a script

1. Pull latest updates before editing
git pull origin main

2. Stage modified files
git add .

3. Commit with a message describing your changes
git commit -m "Your description"

4. Push updates back to main
git push origin main



