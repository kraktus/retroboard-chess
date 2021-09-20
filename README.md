# Retroboard

A chess retrograde move generator based on the python-chess library internals.

## Status

Strong test suite but lack of comparaison of perft result against a trusted source.

## Perft

A very rough perft test on this position gives 672240 moves on depth 3 in \~10s (tested on Apple M1).
<img src="/perft.svg" alt="Perft position" width="250"/>
fen : `q4N2/1p5k/3P1b2/8/6P1/4Q3/3PB1r1/2KR4 b - - 0 1`, with `2PNBRQ` in white pocket, `3NBRQP` in black one, `Q` uncastling and allowing en-passant moves.


## Installation

```
python3 -m venv venv && source venv/bin/activate
pip3 install -r requirements.txt
```