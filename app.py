"""
Sentiment Analysis Web App
Run: python app.py
Open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import csv
import io
import nltk

# Download VADER if needed
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    print("📥 Downloading VADER lexicon...")
    nltk.download("vader_lexicon", quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)
analyzer = SentimentIntensityAnalyzer()


def analyze(text):
    scores = analyzer.polarity_scores(text)
    pos      = round(scores["pos"] * 100, 1)
    neg      = round(scores["neg"] * 100, 1)
    neu      = round(scores["neu"] * 100, 1)
    compound = round(scores["compound"], 4)

    if compound >= 0.05:
        label, emoji = "Positive", "😊"
    elif compound <= -0.05:
        label, emoji = "Negative", "😠"
    else:
        label, emoji = "Neutral", "😐"

    return {
        "label":    label,
        "emoji":    emoji,
        "positive": pos,
        "negative": neg,
        "neutral":  neu,
        "compound": compound,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze-text", methods=["POST"])
def analyze_text():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    return jsonify(analyze(text))


@app.route("/analyze-csv", methods=["POST"])
def analyze_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Please upload a CSV file"}), 400

    content = file.read().decode("utf-8")
    reader  = csv.DictReader(io.StringIO(content))
    fieldnames = reader.fieldnames or []
    rows = list(reader)

    # Auto-detect review column
    col = None
    for c in ["review", "text", "comment", "feedback", "description"]:
        matches = [f for f in fieldnames if c in f.lower()]
        if matches:
            col = matches[0]
            break

    if col is None and fieldnames:
        col = fieldnames[0]

    if col is None:
        return jsonify({"error": "Could not find a review column in the CSV"}), 400

    results = []
    for i, row in enumerate(rows, 1):
        text = row.get(col, "").strip()
        if not text:
            continue
        result = analyze(text)
        result["id"]     = i
        result["review"] = text
        results.append(result)

    if not results:
        return jsonify({"error": "No reviews found in the CSV"}), 400

    total = len(results)
    counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for r in results:
        counts[r["label"]] += 1

    return jsonify({
        "results": results,
        "summary": {
            "total":    total,
            "column":   col,
            "positive": round(counts["Positive"] / total * 100, 1),
            "negative": round(counts["Negative"] / total * 100, 1),
            "neutral":  round(counts["Neutral"]  / total * 100, 1),
            "counts":   counts,
        }
    })


if __name__ == "__main__":
    print("\n🚀 Starting Sentiment Analyzer...")
    print("   Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000)
