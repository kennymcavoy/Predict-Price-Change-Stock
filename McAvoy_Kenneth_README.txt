
1. Running the Script:

Ensure the data files are in the correct directories and the appropriate directories exist as outlined in the file structure section. Then simply run the script using python3, and the output files will be in the output folder. 

Example to run the script:
python3 InterimProject2.py

2. Python and Package Requirements:

Python Version:
Python 3.8.3

Python Packages:
numpy                     1.19.2
pandas                    1.1.3
pandas-datareader         0.9.0 
quandl                    3.5.3
scikit-learn              0.23.2
scipy                     1.5.2
matplotlib                3.3.2

3. File Structure:

|____InterimProject2.py
|____output
| |____small_universe_performance.csv         **
| |____large_universe_performance_top20.csv   **
| |____large_universe_performance.csv         **
|____data
| |____tickers_nasd.csv
| |____tickers.csv
| |____tickers_nyse.csv
| |____stock_dfs
| | |____CSCO.csv
| | |____UAL.csv
| | |____***


** generated files
*** rest of the csv files in stock_dfs

