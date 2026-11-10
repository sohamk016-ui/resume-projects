"""Categorize bank transactions, chart spending, and project savings."""
import argparse
import csv
from collections import defaultdict
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import linear_regression

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CATEGORIES = {
    "Housing": ("rent", "mortgage"),
    "Groceries": ("grocery", "groceries", "supermarket"),
    "Dining": ("coffee", "restaurant", "pizza", "cafe"),
    "Transport": ("fuel", "parking", "bus pass", "rideshare"),
    "Utilities": ("electric", "internet", "water bill"),
}


def money(value):
    amount = Decimal(value.strip().replace("$", "").replace(",", ""))
    if not amount.is_finite() or amount != amount.quantize(Decimal("0.01")):
        raise ValueError("expected an amount with at most two decimal places")
    return amount


def month_index(day):
    return day.year * 12 + day.month - 1


def month_label(index):
    year, month = divmod(index, 12)
    return f"{year:04d}-{month + 1:02d}"


def read_transactions(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"date", "description", "amount"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV requires date, description, and amount columns")
        for row in reader:
            try:
                day = date.fromisoformat(row["date"].strip())
                description = row["description"].strip()
                amount = money(row["amount"])
                if not description:
                    raise ValueError("description is empty")
            except (AttributeError, InvalidOperation, ValueError) as error:
                raise ValueError(f"CSV line {reader.line_num}: invalid date, description, or amount") from error
            yield day, description, amount


def category_for(description):
    description = description.casefold()
    for category, keywords in CATEGORIES.items():
        if any(keyword in description for keyword in keywords):
            return category
    return "Other"


def report(path, output, goal, starting_balance, horizon):
    spending = defaultdict(lambda: defaultdict(Decimal))
    cash_flow = defaultdict(Decimal)
    for day, description, amount in read_transactions(path):
        month = month_index(day)
        cash_flow[month] += amount
        if amount < 0:
            spending[month][category_for(description)] -= amount

    if not cash_flow:
        raise ValueError("CSV contains no transactions")

    months = range(min(cash_flow), max(cash_flow) + 1)
    labels = [month_label(month) for month in months]
    balances = []
    balance = starting_balance
    for month in months:
        balance += cash_flow[month]
        balances.append(float(balance))

    output.mkdir(parents=True, exist_ok=True)
    categories = [*CATEGORIES, "Other"]
    with (output / "monthly_summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["month", *categories, "total_spending", "net_cash_flow"])
        for month in months:
            amounts = [spending[month][category] for category in categories]
            writer.writerow([month_label(month), *amounts, sum(amounts), cash_flow[month]])

    figure, (expense_plot, balance_plot) = plt.subplots(2, 1, figsize=(11, 8), layout="constrained")
    positions = list(range(len(labels)))
    bottom = [0.0] * len(labels)
    for category in categories:
        values = [float(spending[month][category]) for month in months]
        if any(values):
            expense_plot.bar(positions, values, bottom=bottom, label=category)
            bottom = [previous + value for previous, value in zip(bottom, values)]
    if expense_plot.containers:
        expense_plot.legend(ncol=3)
    expense_plot.set(title="Monthly spending by category", ylabel="USD")

    balance_plot.plot(positions, balances, "o-", label="Observed balance")
    balance_plot.axhline(float(goal), color="green", linestyle=":", label="Savings goal")
    last_position = positions[-1]
    if balances[-1] >= float(goal):
        outcome = "The current balance meets the goal."
    elif len(balances) < 2:
        outcome = "A forecast requires at least two calendar months."
    else:
        outcome = f"The forecast does not reach the goal within {horizon} months."

    if len(balances) >= 2:
        slope, _ = linear_regression(positions, balances)
        future = list(range(last_position + 1, last_position + horizon + 1))
        projection = [balances[-1] + slope * (step - last_position) for step in future]
        balance_plot.plot(
            [last_position, *future], [balances[-1], *projection], "--", label="Linear forecast"
        )
        if balances[-1] < float(goal):
            for step, predicted in zip(future, projection):
                if predicted >= float(goal):
                    outcome = f"Projected goal month: {month_label(months[0] + step)}"
                    break
        print(f"Estimated monthly balance change: ${slope:,.2f}")
        labels.extend(month_label(months[0] + step) for step in future)

    for axis, chart_labels in ((expense_plot, labels[:len(months)]), (balance_plot, labels)):
        ticks = range(0, len(chart_labels), max(1, (len(chart_labels) + 9) // 10))
        axis.set_xticks(list(ticks), [chart_labels[tick] for tick in ticks], rotation=30, ha="right")
    balance_plot.set(title="Savings goal forecast", ylabel="USD")
    balance_plot.legend()
    figure.savefig(output / "budget_report.png", dpi=150)
    plt.close(figure)
    print(f"Current balance: ${balance:,.2f}")
    print(outcome)
    print(f"Reports saved in {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV with date, description, amount")
    parser.add_argument("--goal", default="3000")
    parser.add_argument("--starting-balance", default="0")
    parser.add_argument("--forecast-months", type=int, default=6, choices=range(1, 25), metavar="1-24")
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    try:
        goal = money(args.goal)
        if goal <= 0:
            raise ValueError("goal must be positive")
        report(args.csv, args.output, goal, money(args.starting_balance), args.forecast_months)
    except (OSError, csv.Error, InvalidOperation, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
