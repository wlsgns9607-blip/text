import sqlite3
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

def query(sql, args=(), write=False):
    con = sqlite3.connect("iris.db")
    con.row_factory = sqlite3.Row
    cur = con.execute(sql, args)
    rows = [dict(r) for r in cur.fetchall()]
    if write:
        con.commit()
    last = cur.lastrowid
    con.close()
    return rows, last

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/api/data")
def data():
    rows, _ = query("SELECT * FROM iris ORDER BY id")
    return jsonify(rows)

@app.get("/api/runs")
def runs():
    rows, _ = query("SELECT * FROM run ORDER BY id DESC LIMIT 10")
    return jsonify(rows)

@app.post("/api/runs")
def save_run():
    b = request.get_json()
    h = b["history"]
    _, rid = query("INSERT INTO run (lr, epochs, train_acc, test_acc) VALUES (?,?,?,?)",
                   (b["lr"], b["epochs"], h[-1]["acc"], h[-1]["val_acc"]), write=True)
    con = sqlite3.connect("iris.db")
    con.executemany("INSERT INTO epoch_log VALUES (?,?,?,?,?,?)",
        [(rid, x["epoch"], x["loss"], x["acc"], x["val_loss"], x["val_acc"]) for x in h])
    con.commit(); con.close()
    return jsonify({"id": rid})

if __name__ == "__main__":
    app.run(debug=True)
