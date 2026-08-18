from flask import Flask, render_template
from load_data import get_data_summary
from aep_eda import run_eda

app = Flask(__name__)

@app.route("/")
def index():
    # Landing page, no section selected yet
    return render_template("index.html", active="none")

@app.route("/data-loading")
def data_loading():
    """Loads the dataset (server-side) and renders the summary into the page."""
    error = None
    summary = None
    try:
        summary = get_data_summary()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="data-loading",
        summary=summary,
        error=error,
    )

@app.route("/eda")
def eda():
    """Runs exploratory data analysis and renders results."""
    error = None
    eda_output = None
    try:
        eda_output = run_eda()   # call energy EDA function
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "eda.html",
        active="eda",
        results=eda_output,
        error=error,
    )

if __name__ == "__main__":
    app.run(debug=True)
