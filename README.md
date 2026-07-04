# cybersecurity_identify_fraud
Below text is copied verbatim from Vincent Granville's GenAI project assignments - 
_"The case study is about cybersecurity, looking at server data to identify fraud. The amount of fraud is also very small. In this project, most of the heavy work consists of identifying and separating the different parts.

The dataset is located here - https://raw.githubusercontent.com/VincentGranville/Main/main/iot_security.csv. First, you need to do some exploratory analysis to find the peculiarities. For each feature, find the distinct values, their type (category or continuous) and count the multiplicity of each value. Also count the multiplicity of each observation vector when all features are combined. We will split the dataset into three subsets named A, C1 and C2, and remove some outliers in C1 (or at least treat them separately). Simulating observations distributed as in A or C1 is straighforward. For C2, we will use the NoGAN synthesizer.
In the remaining, by observation or observation vector, I mean a full row in the dataset. The project consists of the following steps:

Step 1: Split the data into two subsets A and B. Here A consists of the two big clusters discussed earlier, each containing one observation vector duplicated thousands of times. Also add to A any observation
duplicated at least 4 times. Keep one copy of each unique observation in A, and add one new feature: the observation count, named size. The set B contains all the unique observations except those that are now
in A, with a count (size) for each observation.

Step 2: Create set C as follows. Remove the columns scr port and size from set B. Then keep only one copy of each duplicated observation, and add an extra feature to count the multiplicity attached to each
observation. Finally, split C into C1 and C2, with C1 consisting of observation vectors with multiplicity larger than one, and C2 for the other ones (observation vectors that do not have duplicates). Find outliers
in C1 and remove them.

Step 3: The feature scr port is absent in sets C1 and C2, after step 2. Reconstruct the list of values for scr port with the correct count, separately for C1 and C2, as we will need it in step 4. These satellite
tables are named map C1 and map C2 in the Python code. Double check that all the computations, splitting, mapping, counts, uniques, and aggregation, are correct.

Step 4: Generate synthetic observations for C2, using the NoGAN algorithm described in project 2.1.
Use the following features only:
MMMMM bidirectional syn packets
MMMMM src2dst syn packets
MMMMM application category name
MMMMM application confidence
MMMMM src2dst mean ps
MMMMM src2dst psh packets
MMMMM bidirectional mean ps
MMMMM label
The feature “label” indicates fraud when the value is not 0. Few observations are labeled as non-fraud in C2. How would you proceed to substantially increase the proportion of non-fraud in C2, in the generated
data? How about generating values for the src port feature, based on the map C2 distribution obtained in step 3? Is the distribution in question uniform within its range, between 40,000 and 60,000? If yes, you
could generate uniform values for this feature, possibly different from those actually observed.

Step 5: Think of a general strategy to synthesize observations not just for C2, but for the full original data. Say you want N = 105 synthetic observations. You need to generate n1, n2, n3 observations respectively
for A, C1, C2, with N = n1 + n2 + n3, using a multinomial distribution of parameter [N; p1, p2, p3] where p1, p2, p3 are the the proportions of observations falling respectively in A, C1, and C2.
What are the values of p1, p2, p3? Finally, describe how you would proceed to synthesize observations separately for A, C1, and C2, once the random observation counts n1, n2, n3 have been generated."_
