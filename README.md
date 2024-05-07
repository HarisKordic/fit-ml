# fit-ml
Creating a small AI prediction model. The model needs to predict ECTS credits for a newly added subject based on ECTS credits from previous relevant subjects and grades from them. The model is based on an unsupervised decision tree classification algorithm.

<hr>


Problem: 
Odrediti broj ECTS kredita za novo-uvedeni predmet na četvrtoj godini studija, na osnovu broja ECTS kredita njemu relevantnih predmeta i prosječnim ocjenama iz tih predmeta.
Pod-problemi:
Odrediti kriterij odabira relevantnih predmeta za predmet koji se evaluira. - Rješenje: Napraviti selekciju predmeta za evaluaciju matematičko-tehnički orijentisanih predmeta s prve, druge i treće godine studija Fakulteta Informacijskih Tehnologija - FIT.
Odabrati adekvatan klasifikacijski algoritam za predikciju - Rješenje: Decision tree
Odlučiti između supervised ili unsupervised metode - Rješenje: Unsupervised
Odlučiti se za regresijski ili klasifikacijski pristup - Rješenje: Klasifikacija
Odrediti prosječne ocjene iz navedenih predmeta. - Rješenje: Iz raw podataka napraviti selekciju relevantnih predmeta te za svaki relevantni predmet uzeti ocjene.
Odrediti broj klasa i same klase za decision tree model - Rješenje: Visoka, srednja i niska klasa; 8 plus ECTS kredita, od 6-8, do 6 ECTS kredita, respektivno
Ili napraviti da output bude samo broj ECTS kredita, ne da bude klasa.

<hr>

Predmeti po kodovima:

P-175 Programiranje I
P-176 Programiranje II
P-177 Programiranje III
P-150 Razvoj softvera I
P-157 Razvoj softvera II

Matematike i ai ml

Decision trees logika:

- Sequence of if-else questions
- Consists of hierarchy of nodes. Each node raises a question or prediction.
- Root node : No parent
- Internal node : Has parent, has children
- Leaf node : Has no children. It is where predictions are made
- Goal : Search for pattern to produce purest leaves. Each leaf contains a pattern for one dominant label.
- Information Gain : At each node, find the split point for each feature for which we get maximum correct pure split of the data. When information gain = 0, we could say that our goal is achieved, the pattern is captured, and this is a leaf node. Otherwise keep splitting it (We can stop it by specifying maximum depth of recursion split). 
- Measure of impurity in a node:
    - Gini index: For classification
    - Entropy: For classification
    - MSE : For regression
- capture non-linear relationship between features and labels/ real values
- Do not require feature scaling
- At each split, only one feature is involved
- Decision region : Feature space where instances are assigned to a label / value
- Decision Boundary : Surface that separates different decision regions
- Steps of building a decision tree:
    1. Choose an attribute (column) of dataset
    2. Calculate the significance of that attribute when splitting the data with Entropy.
        A good split has less Entropy (disorder / randomness). 
    3. Find the best attribute that has most significance and use that attribute
    	to split the data
    4. For each branch, repeat the process (Recursive partitioning) for best 
    	information gain (The path that gives the most information using entropy).
- Limitations:
    - Can only produce orthogonal decision boundaries
    - Sensitive to small variations in training set
    - High variance overfits the model
- Solution : Ensemble learning
    - Train different models on same dataset
    - Let each model make its prediction
    - Aggregate predictions of individual models (eg: hard-voting)
    - One model's weakness is covered by another model's strength in that particular task
    - Final model is combination of models that are skillful in different ways


