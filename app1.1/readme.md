# Linux Problem 
1. dont forget to change the folder owner to user not root, this problem can lead to permission error when tried to run teh app

# Major Update
1. Translate all the text to Bahasa Indonesia on the main app -> affected in app.py
2. Unactivated detik.com as news source because of some trouble-> affected in config.py and app.py and deleted detik.py from previous version

# Minor Update
1. Reduce default max article per source to 100 -> affected in app.py 
2. Delete model to only use 1 model -> affected in config.py and app.py

3. Add help description on the start and date -> affected in app.py
4. Limit the Start date to only 1 Januari 2020 and the end date will be limited to user "current date" -> affected in app.py
5. Add timer for the whole proses -> app.py and run_analyze.py (for the news source)
6. Increase Model Batch Size for inference to 128 -> config.py