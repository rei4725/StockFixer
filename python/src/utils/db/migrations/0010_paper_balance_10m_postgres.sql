-- 紙の口座の元手を 100 万円から 1000 万円にする一度きりの入金 #1877
-- 持ち株と約定は残し、現金に 900 万円を足す。入金の出来事を paper_cash_events に残す
CREATE TABLE IF NOT EXISTS paper_cash_events (
    id          BIGSERIAL PRIMARY KEY,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    amount      DOUBLE PRECISION NOT NULL,
    reason      TEXT NOT NULL
);

INSERT INTO paper_cash_events (amount, reason)
VALUES (9000000.0, 'paper_balance_10m #1877');

UPDATE paper_balance SET balance = balance + 9000000.0;
