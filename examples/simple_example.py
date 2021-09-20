#!/usr/local/bin/python3
#coding: utf-8


from retroboard import RetrogradeBoard, UnMove

rboard = RetrogradeBoard("q4N2/1p5k/3P1b2/8/6P1/4Q3/3PB1r1/2KR4 b - - 0 1",
                    pocket_w="2PNBRQ",
                    pocket_b="3NBRQP",
                    allow_ep=True, 
                    uncastling_rights="Q")
print(rboard.is_valid())
# True
rboard.pp()
# ♛ . . . . ♘ . .
# . ♟ . . . . . ♚
# . . . ♙ . ♝ . .
# . . . . . . . .
# . . . . . . ♙ .
# . . . . ♕ . . .
# . . . ♙ ♗ . ♜ .
# . . ♔ ♖ . . . .
print(rboard.legal_unmoves)
# <LegalUnMoveGenerator at 0x1008c4b50 (f8d7, Nf8d7, Bf8d7, Rf8d7, Qf8d7, f8g6, Nf8g6, Bf8g6, Rf8g6, Qf8g6, f8e6, Nf8e6, Bf8e6, Rf8e6, Qf8e6, Uf8f7, UNf8e7, UBf8e7, URf8e7, UQf8e7, UNf8g7, UBf8g7, URf8g7, UQf8g7, c1e1)>
rboard.retropush(UnMove.from_retro_uci("Uf8f7"))
unmove = UnMove.from_retro_uci("Qg2h2")
print(rboard.is_retrolegal(unmove))
# True
rboard.retropush(unmove)
rboard.retropush(UnMove.from_retro_uci("c1e1"))
rboard.pp()
# ♛ . . . . . . .
# . ♟ . . . ♙ . ♚
# . . . ♙ . ♝ . .
# . . . . . . . .
# . . . . . . ♙ .
# . . . . ♕ . . .
# . . . ♙ ♗ . ♕ ♜
# ♖ . . . ♔ . . .
rboard.retropop()
# c1e1
rboard.retropop()
# Qg2h2
rboard.retropop()
# Uf8f7
rboard.pp()
# ♛ . . . . ♘ . .
# . ♟ . . . . . ♚
# . . . ♙ . ♝ . .
# . . . . . . . .
# . . . . . . ♙ .
# . . . . ♕ . . .
# . . . ♙ ♗ . ♜ .
# . . ♔ ♖ . . . .