# UQ-TSS
Welcome to the repsitory for UQ-TSS!

UQ-TSS is a general framework for uncertainty quantification (UQ) of time series segmentation (TSS). UQ-TSS is designed to quantify uncertainty of any segmentation from any  given TSS method and time series (TS). UQ-TSS is an ensemble learning approach that estimates probability distributions of TSS output components: The CP presence and the CP location. UQ-TSS characterizes these distributions to yield final CP predictions and measure uncertainty of the entire segmentation. 

UQ-TSS provides new measures for TSS evaluation that do not rely on ground truth. UQ-TSS can help optimize hyper-parameters, create different interpretations of a segmentation, and refine segmentation results based on its uncertainty measures. 

## Datasets
The datasets from our experiments can be found from the following links.
PAMAP2: https://drive.google.com/drive/folders/1ECwHJetl8EPRkQSMD-rLuWimJ5kMp8qW

The 32 TS Repository can be found from the following authors: https://www.cs.ucr.edu/~eamonn/Segmentation_ICDM.pdf 

Properties of 32 TS Repository:
| Dataset Name | Length | Number of changepoints |
| ------------ | ------ | ---------------------- |
| Cane_100_2345.txt | 5340 | 1 |
| DutchFactory_24_2184.txt | 8761 | 1 |
| EEGRat2_10_1000.txt | 2000 | 1 |
| EEGRat_10_1000.txt | 2000 | 1 |
| Fetal2013_70_6000_12000.txt | 18000 | 2 |
| GrandMalSeizures2_10_4550.txt | 10433 | 1 |
| GrandMalSeizures_10_8200.txt | 18432 | 1 |
| GreatBarbet1_50_1900_3700.txt | 4700 | 2 |
| GreatBarbet2_50_1900_3700.txt | 4700 | 2 |
| InsectEPG1_50_3802.txt | 17001 | 1 |
| InsectEPG2_50_1800.txt | 10001 | 1 |
| InsectEPG3_50_1710.txt | 7001 | 1 |
| InsectEPG4_50_3160.txt | 19001 | 1 |
| NogunGun_150_3000.txt | 7383 | 1 |
| PigInternalBleedingDatasetAirwayPressure_400_7501.txt | 14973 | 1 |
| PigInternalBleedingDatasetArtPressureFluidFilled_100_7501.txt | 14973 | 1 |
| PigInternalBleedingDatasetCVP_100_7501.txt | 14973 | 1 |
| Powerdemand_12_4500.txt | 7682 | 1 |
| PulsusParadoxusECG1_30_10000.txt | 17521 | 1 |
| PulsusParadoxusECG2_30_10000.txt | 17521 | 1 |
| PulsusParadoxusSP02_30_10000.txt | 17521 | 1 |
| RoboticDogActivityX_60_8699.txt | 12699 | 1 |
| RoboticDogActivityY_60_10699.txt | 14699 | 1 |
| RoboticDogActivityY_64_4000.txt | 11000 | 1 |
| SimpleSynthetic_125_3000_5000.txt | 8001 | 2 |
| SuddenCardiacDeath1_25_6200_7600.txt | 12000 | 2 |
| SuddenCardiacDeath2_25_3250.txt | 12001 | 1 |
| SuddenCardiacDeath3_25_3250.txt | 12001 | 1 |
| TiltABP_210_25000.txt | 40000 | 1 |
| TiltECG_200_25000.txt | 40000 | 1 |
| WalkJogRun1_80_3800_6800.txt | 10001 | 2 |
| WalkJogRun2_80_3800_6800.txt | 10001 | 2 |

![Historgram of TS Lenths](./Images/lengths.png)
![Historgram of TS Cps](./Images/Cps.png)

## Running UQ-TSS
We provide Jupyter Notebooks demonstrating UQ-TSS with PELT, FLOSS, CLaSP, and BOCPD. Please look in the folder "Jupyter Notebooks" 
