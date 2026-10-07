import csv, sqlite3
con = sqlite3.connect("iris.db")
con.executescript(open("schema.sql", encoding="utf-8").read())
with open("data/iris.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
for i, r in enumerate(rows):
    # 품종별로 50개씩 정렬돼 있으므로 5번째마다 테스트로 → 품종당 10개씩 균등 분할
    con.execute("INSERT INTO iris VALUES (?,?,?,?,?,?,?)",
        (i + 1, r["sepal_length"], r["sepal_width"], r["petal_length"],
         r["petal_width"], r["species"], 1 if i % 5 == 0 else 0))
con.commit()
print(f"{len(rows)}개 행 저장 완료")
