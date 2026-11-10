# Resume projects

Python 3.11 or newer. The bundled CSVs contain fictional examples for trying the programs.

## Run in VS Code on Windows

Extract the ZIP, open the `resume-projects` folder in VS Code, and open **Terminal > New Terminal**. Run these commands in PowerShell. Install Python if the `py` launcher is unavailable.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe SmartBudget/smartbudget.py SmartBudget/data/sample_transactions.csv --goal 4000
.\.venv\Scripts\python.exe GuardianAI/guardianai.py train GuardianAI/data/sample_emails.csv
.\.venv\Scripts\python.exe GuardianAI/guardianai.py predict --text "Urgent: verify your account password immediately."
```

On macOS or Linux, create the environment with `python3 -m venv .venv` and replace each `.\.venv\Scripts\python.exe` with `./.venv/bin/python`.

## SmartBudget

SmartBudget reads a UTF-8 CSV with `date,description,amount` headers. Use dates in `YYYY-MM-DD` format and signed USD amounts. Expenses are negative and deposits are positive. Rename or map your bank's headers and signs to this schema before running it. Use a single account and exclude internal transfers or duplicate rows where appropriate.

The demo generates `output/monthly_summary.csv` and `output/budget_report.png`. It reports a $2,510.50 balance and projects reaching the $4,000 goal in July 2025. Pass `--starting-balance` for a balance carried in from before the export and `--forecast-months` to set the forecast horizon. The forecast fits a linear trend to month-end balances and extends that slope from the latest observed balance. Months absent from the CSV count as zero activity.

## GuardianAI

GuardianAI reads a UTF-8 CSV with `text,label` headers and labels of `legitimate` or `phishing`. It removes duplicate messages, fits TF-IDF and logistic regression on a training split, and reports metrics on a separate holdout. The sample command saves a model to `output/guardianai.joblib`. For a message saved as plain text, use `predict --file message.txt`. Load only model files you created or trust.

The included 24 synthetic emails demonstrate the workflow. Evaluating performance on real email requires a representative labeled dataset and an independent evaluation set. The original datasets and historical results are outside the scope of this implementation.
