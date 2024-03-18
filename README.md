# Insole Plantar Pressure to Identify PD

This project was developed to utilize insole plantar pressure data to differentiate between individuals with and without Parkinson's disease (PD) and PD fallers from PD non-fallers. 
Insole plantar pressure data was collected from 111 people (67 without PD, 44 with PD) during six different balance tasks. The tasks comprised of three static tasks and three active tasks. The insole plantar pressure was post processed to develop 60 features per task. These features can be found in data/df_avg_s (for average COP during static tasks), data/df_avg_a (for average active tasks), data/df_asym_s (for asymmetry of COP during static tasks), and data/df_asym_a (for asymmetry of COP active tasks).


The labels for each sample (0: control, 1: PD) can be found in the first column ["group"] of data/df_y.csv.
The labels which indicate whether an individual with a faller (0: nonfaller, 1: faller) can be found in the second column ["faller"] of data/df_y.csv.

The balance tasks and their acronyms are outlined below

STATIC BALANCE TASKS

QSEO:quiet stance eyes open
QSEC:quiet stance eyes closed
QSOF:quiet stance one foot

ACTIVE BALANCE TASKS

GAIT: gait
FR:functional reach
BO:bend over


The features found in data/df_avg_s.csv data/df_avg_a.csv, data/df_asym_s.csv, and data/df_asym_a.csv were calculated from raw insole plantar pressure data collected as part of this study. The algorithms for calculating these features were from: https://github.com/Jythen/code_descriptors_postural_control
