DROP TABLE IF EXISTS epoch_log;
DROP TABLE IF EXISTS run;
DROP TABLE IF EXISTS iris;

CREATE TABLE iris (
  id INTEGER PRIMARY KEY,
  sepal_length REAL, sepal_width REAL,
  petal_length REAL, petal_width REAL,
  species TEXT NOT NULL,
  is_test INTEGER NOT NULL DEFAULT 0   -- 1 = 테스트(20%)
);
CREATE TABLE run (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  lr REAL, epochs INTEGER, train_acc REAL, test_acc REAL
);
CREATE TABLE epoch_log (
  run_id INTEGER REFERENCES run(id),
  epoch INTEGER, loss REAL, acc REAL, val_loss REAL, val_acc REAL,
  PRIMARY KEY (run_id, epoch)
);
