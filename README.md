## Modeling Co-Decisions to Understand How Pairs of Actors Differ/Agree in their Decision-Making

Using data at the individual actor-level on their decisions as well as descriptive features, we model pairs of actors making the same or different decisions, and the explanation offered by what we know of those actors or their behavior (as features). 

This generalizes the co-voting modeling framework in Ringe et al (2013). 


### Data Preparation
---

**The data directory must follow the format laid down below; please see the data/example/ directory provided as a guide to understand the format.**

Each data directory must have one CSV file named **decisions.csv**, and one subdirectory named **features/**. Note that none of the csv files are expected to have any headers/column names - please create files without column names and in the format described below (data/example/ provides examples of file formats that can be used). 

**decisions.csv**: This contains the categorical *decisions* made by every *actor* on each of multiple *items* - the rows corresponding to each individual actor, the first column contains IDs of each actor, and the subsequent columns all contain the integer value of all the decisions made. 

**features/**: Contains multiple CSV files where each CSV file corresponds to a particular feature and its values for each actor. Features can be of two types: *scalar* (one value quantifiying each actor) and *vector* (a vector of more than one values quantifying each actor). NOTE: Name each of these feature files appropriately, since they will be used to interpret the results (the name of the file is used as the feature name). 


### Running the codes
---

Python version required: 3.7+

Please ensure all required Python libraries are installed (presented in `requirements.txt`). 

**Step 1:** Run code to use individual actor-level data and prepare pairwise data file: `python create_pairwise_data_file.py --data <path-to-data-directory> --output <path-to-directory-for-storing-output-file>`

To work with out example data, run: `python create_pairwise_data_file.py --data data/example/ --output data/example/output/`

To view all the arguments that can be passed to the script and what they entail, run: `python create_pairwise_data_file.py -h`

**Step 2:** Run the mixed effect models implement in our R script: `Rscript mixed_effects_model.R <path-to-decisions-CSV-file> <path-to-pairwise-data-file-created-in-step-1> <output-filepath-for-storing-results-of-the-model>` 

To work with our example data, following on the step 1 command for our example, run: `Rscript mixed_effects_model.R data/example/decisions.csv data/example/output/pairwise_data.csv data/example/output/results.txt`

**NOTE: A walkthrough of what the above code does is offered using our example data in the iPython notebook: example.ipynb.**


### Reference
---

If you use this repository, please cite it as: **Goel, P. (2023). Modeling Co-Decisions to Understand How Pairs of Actors Differ/Agree in their Decision-Making (Version 1.0.0) [Computer software]. https://github.com/Pranav-Goel/co-decisions**. In addition, cite the co-voting work by Ringe et al as: **Ringe, Nils, Jennifer Nicoll Victor, and Justin H. Gross. "Keeping your friends close and your enemies closer? Information networks in legislative politics." British Journal of Political Science 43, no. 3 (2013): 601-628.**, since we note that the mixed effects modeling approach and some choices made in this repository are derived from that work (our implementation offers a more generalized framework to conduct experiments like the ones conducted in Ringe et al (2013)).  

BibTeX for our software: 
`
@software{Goel_Modeling_Co-Decisions_to_2023,
author = {Goel, Pranav},
month = {2},
title = {{Modeling Co-Decisions to Understand How Pairs of Actors Differ/Agree in their Decision-Making}},
url = {https://github.com/Pranav-Goel/co-decisions},
version = {1.0.0},
year = {2023}
}
`
