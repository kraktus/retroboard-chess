#!/usr/local/bin/python3
#coding: utf-8

"""
A chess retrograde move generator
"""

import chess
import unittest

from retroboard import RetrogradeBoard, UnMove, RetrogradeBoardPocket, NotInPocket, TooManyNumberForPocket

###########
#Constants#
###########



#########
#Classes#
#########

class TestUnMove(unittest.TestCase):

    def test_uci_move_from_retro_uci(self):
        unmove = UnMove.from_retro_uci("e2e4")
        self.assertEqual(unmove.from_square, chess.E2)
        self.assertEqual(unmove.to_square, chess.E4)
        self.assertEqual(unmove.unpromotion, False)
        self.assertEqual(unmove.uncapture, None)

    def test_to_retro_uci(self):
        self.assertEqual(UnMove.from_retro_uci("a1a8").retro_uci(), "a1a8")
        self.assertEqual(UnMove.from_retro_uci("Qa1a8").retro_uci(), "Qa1a8")
        self.assertEqual(UnMove.from_retro_uci("Ua1a2").retro_uci(), "Ua1a2")
        self.assertEqual(UnMove.from_retro_uci("Ua1b2").retro_uci(), "Ua1b2")
        self.assertEqual(UnMove.from_retro_uci("Ec3d4").retro_uci(), "Ec3d4")

    def test_eq_(self):
        """test `to_tuple` and `__eq__`"""
        # uci
        u_1 = UnMove.from_retro_uci("a1a8")
        u_2 = UnMove.from_retro_uci("a1a8")
        self.assertEqual(u_1, u_2)
        # uncapture
        u_3 = UnMove.from_retro_uci("Qa1a8")
        u_4 = UnMove.from_retro_uci("Qa1a8")
        self.assertEqual(u_3, u_4)
        # unpromote
        u_5 = UnMove.from_retro_uci("Ua1a2")
        u_6 = UnMove.from_retro_uci("Ua1a2")
        self.assertEqual(u_5, u_6)
        # unpromote & uncapture
        u_7 = UnMove.from_retro_uci("Ua1b2")
        u_8 = UnMove.from_retro_uci("Ua1b2")
        self.assertEqual(u_7, u_8)
        u_9 = UnMove.from_retro_uci("Ef3e4")
        u_10 = UnMove.from_retro_uci("Ef3e4")
        self.assertEqual(u_9, u_10)

    def test_mirror(self):
        # uci
        u_1 = UnMove.from_retro_uci("a1a8")
        u_2 = UnMove.from_retro_uci("a8a1")
        self.assertEqual(u_1.mirror(), u_2)
        # uncapture
        u_3 = UnMove.from_retro_uci("Qa8a1")
        u_4 = UnMove.from_retro_uci("Qa1a8")
        self.assertEqual(u_3.mirror(), u_4)
        # unpromote
        u_5 = UnMove.from_retro_uci("Ua1a2")
        u_6 = UnMove.from_retro_uci("Ua8a7")
        self.assertEqual(u_5.mirror(), u_6)
        # unpromote & uncapture
        u_7 = UnMove.from_retro_uci("Ua1b2")
        u_8 = UnMove.from_retro_uci("Ua8b7")
        self.assertEqual(u_7.mirror(), u_8)
        u_9 = UnMove.from_retro_uci("Ef3e4")
        u_10 = UnMove.from_retro_uci("Ef6e5")
        self.assertEqual(u_9.mirror(), u_10)

    def test_uncapture_retro_uci(self):
        for piece in "NBRQ":
            with self.subTest(piece=piece):
                unmove = UnMove.from_retro_uci(f"{piece}a1a8")
                self.assertEqual(unmove.from_square, chess.A1)
                self.assertEqual(unmove.to_square, chess.A8)
                self.assertEqual(unmove.uncapture, chess.Piece.from_symbol(piece).piece_type)
                self.assertEqual(unmove.unpromotion, False)

    def test_illegal_uncapture_retro_uci(self):
        self.assertRaises(ValueError, UnMove.from_retro_uci,"Ka2a6")

    def test_unpromotion_retro_uci(self):
        unmove = UnMove.from_retro_uci("Uh8h7")
        self.assertEqual(unmove.from_square, chess.H8)
        self.assertEqual(unmove.to_square, chess.H7)
        self.assertEqual(unmove.unpromotion, True)
        self.assertEqual(unmove.uncapture, None)

    def test_en_passant_retro_uci(self):
        unmove = UnMove.from_retro_uci("Eg6f5")
        self.assertEqual(unmove.from_square, chess.G6)
        self.assertEqual(unmove.to_square, chess.F5)
        self.assertEqual(unmove.unpromotion, False)
        self.assertEqual(unmove.uncapture, None)
        self.assertEqual(unmove.en_passant, True)


class TestRetrogradeBoardPocket(unittest.TestCase):

    def test_init(self):
        RetrogradeBoardPocket()
        RetrogradeBoardPocket("PNBRQ")
        for i in range(9):
            RetrogradeBoardPocket(f"PNBRQ{i}")
        self.assertRaises(TooManyNumberForPocket, RetrogradeBoardPocket, "PNBRQ12")


    def test_eq(self):
        self.assertEqual(RetrogradeBoardPocket("PQP"), RetrogradeBoardPocket("PPQ"))
        self.assertEqual(RetrogradeBoardPocket(), RetrogradeBoardPocket())
        self.assertNotEqual(RetrogradeBoardPocket("2NBRQ"), RetrogradeBoardPocket("NBRQ6"))

class TestRetrogradeBoard(unittest.TestCase):

    def test_mirror(self):
        board = RetrogradeBoard("r1bq1r2/pp2n3/4N2k/3pPppP/1b1n2Q1/2N5/PP3PP1/R1B1K2R w KQ g6 0 15",pocket_w="QRRP")
        mirrored = RetrogradeBoard("r1b1k2r/pp3pp1/2n5/1B1N2q1/3PpPPp/4n2K/PP2N3/R1BQ1R2 b kq g3 0 15",pocket_b="QRRP")
        self.assertEqual(board.mirror(), mirrored)
        self.assertEqual(board.mirror().mirror(), board)

    def test_eq(self):
        self.assertEqual(RetrogradeBoard(), RetrogradeBoard())
        self.assertNotEqual(RetrogradeBoard(), RetrogradeBoard(pocket_w="Q"))        
        self.assertNotEqual(RetrogradeBoard(pocket_b="Q"), RetrogradeBoard(pocket_w="Q")) 

    def test_simple_retropush_unmove(self):
        retrogradeboard = RetrogradeBoard(fen="r3k3/8/8/8/8/8/8/4K3 w - - 0 1")
        unmove = UnMove.from_retro_uci("a8a2")
        retrogradeboard.retropush(unmove)
        retrogradeboard_2 = RetrogradeBoard(fen="4k3/8/8/8/8/8/r7/4K3 b - - 0 2")
        self.assertTrue(retrogradeboard.is_valid())
        self.assertEqual(retrogradeboard, retrogradeboard_2)

    def test_uncapture_retropush_unmove(self):
        for piece in "NBRQ":
            with self.subTest(piece=piece):
                retrogradeboard = RetrogradeBoard(fen="r3k3/8/8/8/8/8/8/4K3 w - - 0 1", pocket_w=piece)
                unmove = UnMove.from_retro_uci(f"{piece}a8a2")
                retrogradeboard.retropush(unmove)
                retrogradeboard_2 = RetrogradeBoard(fen=f"{piece}3k3/8/8/8/8/8/r7/4K3 b - - 0 2")
                self.assertTrue(retrogradeboard.is_valid())
                self.assertEqual(retrogradeboard, retrogradeboard_2)

    def test_illegal_uncapture_retropush(self):
        """An uncapture of a piece which is not in the pocket of the board"""
        for piece in "NBRQ":
            with self.subTest(piece=piece):
                retrogradeboard = RetrogradeBoard()
                unmove = UnMove.from_retro_uci(f"{piece}g8f6")
                self.assertRaises(NotInPocket, retrogradeboard.retropush, unmove)

    def test_illegal_unpromotion_retropush(self):
        """An uncapture of a piece which is not in the pocket of the board"""
        retrogradeboard = RetrogradeBoard("1R6/7k/8/8/8/8/8/1K6 b - - 0 1")
        unmove = UnMove.from_retro_uci(f"Ub8b7")
        retrogradeboard.retropush(unmove)
        self.assertEqual(retrogradeboard.pockets[chess.WHITE].nb_unpromotion, -1)

    def test_unpromotion_retropush_repropop(self):
        for i in range(1,9):
            with self.subTest(number_of_piece_than_can_unpromote=i):
                retrogradeboard = RetrogradeBoard("1R6/7k/8/8/8/8/8/1K6 b - - 0 1", pocket_w=str(i))
                unmove = UnMove.from_retro_uci(f"Ub8b7")
                retrogradeboard.retropush(unmove)
                self.assertEqual(retrogradeboard.pockets[chess.WHITE].get_nb_unpromotion(), i -1)

    def test_unpromotion_retropush_unmove(self):
        retrogradeboard = RetrogradeBoard(fen="r3k3/8/8/8/8/8/8/4K3 w - - 0 1", pocket_b="1")
        unmove = UnMove.from_retro_uci("Ua8a7")
        retrogradeboard.retropush(unmove)
        retrogradeboard_2 = RetrogradeBoard(fen="4k3/p7/8/8/8/8/8/4K3 b - - 0 2")
        self.assertTrue(retrogradeboard.is_valid())
        self.assertEqual(retrogradeboard, retrogradeboard_2)

    def test_en_passant_retropush_unmove(self):
        retrogradeboard = RetrogradeBoard(fen="rnbqkbnr/pppp1ppp/8/8/8/5p2/PPPPP1PP/RNBQKBNR w KQkq - 0 1", pocket_w="P")
        unmove = UnMove.from_retro_uci("Ef3e4")
        retrogradeboard.retropush(unmove)
        retrogradeboard_2 = RetrogradeBoard(fen="rnbqkbnr/pppp1ppp/8/8/4pP2/8/PPPPP1PP/RNBQKBNR b KQkq - 0 2")
        self.assertTrue(retrogradeboard.is_valid())
        self.assertEqual(retrogradeboard, retrogradeboard_2)

    def test_unpromotion_and_uncapture_retro_unmove(self):
        for piece in "NBRQ":
            with self.subTest(piece=piece):
                retrogradeboard = RetrogradeBoard(fen="r3k3/8/8/8/8/8/8/4K3 w - - 0 1",pocket_b="1",pocket_w=piece)
                unmove = UnMove.from_retro_uci(f"U{piece}a8b7")
                retrogradeboard.retropush(unmove)
                retrogradeboard_2 = RetrogradeBoard(fen=f"{piece}3k3/1p6/8/8/8/8/8/4K3 b - - 0 2")
                self.assertTrue(retrogradeboard.is_valid())
                self.assertEqual(retrogradeboard, retrogradeboard_2)

    def test_retropop_after_simple_retropush(self):
        retrogradeboard = RetrogradeBoard(fen="r3k3/8/8/8/8/8/8/4K3 w - - 0 1",pocket_b="P",pocket_w="QRR")
        unmove = UnMove.from_retro_uci("a8a2")
        retrogradeboard.retropush(unmove)
        retrogradeboard.retropop()
        retrogradeboard_2 = RetrogradeBoard(fen="r3k3/8/8/8/8/8/8/4K3 w - - 0 1",pocket_b="P",pocket_w="QRR")
        self.assertTrue(retrogradeboard.is_valid())
        self.assertEqual(retrogradeboard, retrogradeboard_2)

    def test_retropop_after_simple_retropush_with_different_nb_unpromotion(self):
        retrogradeboard = RetrogradeBoard(fen="r3k3/8/8/8/8/8/8/4K3 w - - 0 1",pocket_b="1",pocket_w="QRR")
        unmove = UnMove.from_retro_uci("a8a2")
        retrogradeboard.retropush(unmove)
        retrogradeboard.retropop()
        retrogradeboard_2 = RetrogradeBoard(fen="r3k3/8/8/8/8/8/8/4K3 w - - 0 1",pocket_b="6",pocket_w="QRR")
        self.assertTrue(retrogradeboard.is_valid())
        self.assertNotEqual(retrogradeboard, retrogradeboard_2)

    def test_retropop_after_uncapture(self):
        retrogradeboard = RetrogradeBoard(fen="r3k3/8/8/8/8/8/8/4K3 w - - 0 1",pocket_b="P",pocket_w="QRR")
        unmove = UnMove.from_retro_uci("Qa8a2")
        retrogradeboard.retropush(unmove)
        retrogradeboard.retropop()
        retrogradeboard_2 = RetrogradeBoard(fen="r3k3/8/8/8/8/8/8/4K3 w - - 0 1",pocket_b="P",pocket_w="QRR")
        self.assertTrue(retrogradeboard.is_valid())
        self.assertEqual(retrogradeboard, retrogradeboard_2)

    def test_retropop_after_en_passant(self):
        retrogradeboard = RetrogradeBoard(fen="rnbqkbnr/pppp1ppp/8/8/8/5p2/PPPPP1PP/RNBQKBNR w KQkq - 0 1", pocket_w="P", pocket_b="QQ")
        unmove = UnMove.from_retro_uci("Ef3e4")
        retrogradeboard.retropush(unmove)
        retrogradeboard.retropop()
        retrogradeboard_2 = RetrogradeBoard(fen="rnbqkbnr/pppp1ppp/8/8/8/5p2/PPPPP1PP/RNBQKBNR w KQkq - 0 1", pocket_w="P", pocket_b="QQ")
        self.assertTrue(retrogradeboard.is_valid())
        self.assertEqual(retrogradeboard, retrogradeboard_2)

    def test_retropop_after_unpromotion_retropush(self):
        retrogradeboard = RetrogradeBoard(fen="r3k3/8/8/8/8/8/8/4K3 w - - 0 1",pocket_b="1")
        unmove = UnMove.from_retro_uci("Ua8a7")
        retrogradeboard.retropush(unmove)
        retrogradeboard.retropop()
        retrogradeboard_2 = RetrogradeBoard(fen="r3k3/8/8/8/8/8/8/4K3 w - - 0 1",pocket_b="1")
        self.assertTrue(retrogradeboard.is_valid())
        self.assertEqual(retrogradeboard, retrogradeboard_2)

    def test_is_unmove_retro_zeroing(self):
        rboard = RetrogradeBoard()
        dic = {"a1a2": False, "Ue8e7": True, "Rd4h4": True, "UNe1d2": True, "b1g1": False}
        for retro_uci, res in dic.items():
            with self.subTest(retro_uci=retro_uci):
                unmove = UnMove.from_retro_uci(retro_uci)
                self.assertEqual(rboard.is_retro_zeroing(unmove), res)
        
    def test_simple_pseudo_legal_unmove_king_knight(self):
        retrogradeboard = RetrogradeBoard(fen="2k5/8/8/8/8/8/8/K6N b - - 0 1")
        self.assertTrue(retrogradeboard.is_valid())
        unmoves = ["a1a2", "a1b2", "a1b1", "h1f2", "h1g3"]
        unmoves_set = set([UnMove.from_retro_uci(i) for i in unmoves])
        unmoves_set_b = set([UnMove.from_retro_uci(i).mirror() for i in unmoves])
        self.assertEqual(unmoves_set, set(retrogradeboard.generate_pseudo_legal_unmoves()))
        self.assertEqual(unmoves_set_b, set(retrogradeboard.mirror().generate_pseudo_legal_unmoves()))

    def test_simple_pseudo_legal_unmove_queen(self):
        retrogradeboard = RetrogradeBoard(fen="3k4/8/8/8/8/4b3/nn6/Kn2b1Q1 b - - 0 1")
        self.assertTrue(retrogradeboard.is_valid())
        unmoves = ["g1g2", "g1g3", "g1g4", "g1g5", "g1g6", "g1g7", "g1g8", "g1f1", "g1f2", "g1h1", "g1h2"]
        unmoves_set = set([UnMove.from_retro_uci(i) for i in unmoves])
        unmoves_set_b = set([UnMove.from_retro_uci(i).mirror() for i in unmoves])
        self.assertEqual(unmoves_set, set(retrogradeboard.generate_pseudo_legal_unmoves()))
        self.assertEqual(unmoves_set_b, set(retrogradeboard.mirror().generate_pseudo_legal_unmoves()))

    def test_simple_pseudo_legal_unmove_no_take(self):
        """It's illegal to take a piece"""
        retrogradeboard = RetrogradeBoard(fen="2k5/8/8/8/8/6n1/nn3n2/Kn5N b - - 0 1")
        self.assertTrue(retrogradeboard.is_valid())
        self.assertFalse(any(retrogradeboard.generate_pseudo_legal_unmoves()))

    def test_pseudo_legal_unmove_simple_pawn_unmove(self):
        retrogradeboard = RetrogradeBoard(fen="2k5/8/8/5P2/8/8/nn6/Kn6 b - - 0 1")
        self.assertTrue(retrogradeboard.is_valid())
        unmoves = ["f5f4"]
        unmoves_set = set([UnMove.from_retro_uci(i) for i in unmoves])
        unmoves_set_b = set([UnMove.from_retro_uci(i).mirror() for i in unmoves])
        self.assertEqual(unmoves_set, set(retrogradeboard.generate_pseudo_legal_unmoves()))
        self.assertEqual(set(retrogradeboard.pseudo_legal_unmoves), set(retrogradeboard.generate_pseudo_legal_unmoves()))
        self.assertEqual(unmoves_set_b, set(retrogradeboard.mirror().generate_pseudo_legal_unmoves()))

    def test_pseudo_legal_unmove_simple_and_double_pawn_unmove(self):
        retrogradeboard = RetrogradeBoard(fen="2k5/8/8/8/5P2/8/nn6/Kn6 b - - 0 1")
        self.assertTrue(retrogradeboard.is_valid())
        unmoves = ["f4f3", "f4f2"]
        unmoves_set = set([UnMove.from_retro_uci(i) for i in unmoves])
        unmoves_set_b = set([UnMove.from_retro_uci(i).mirror() for i in unmoves])
        self.assertEqual(unmoves_set, set(retrogradeboard.generate_pseudo_legal_unmoves()))
        self.assertEqual(set(retrogradeboard.pseudo_legal_unmoves), set(retrogradeboard.generate_pseudo_legal_unmoves()))
        self.assertEqual(unmoves_set_b, set(retrogradeboard.mirror().generate_pseudo_legal_unmoves()))

    def test_pseudo_legal_unmove_no_pawn_unmove(self):
        """Pawn can't unmove when on the 2nd (for white) or 7th rank"""
        retrogradeboard = RetrogradeBoard(fen="1k6/8/8/8/8/8/3P2nn/6nK b - - 0 1")
        self.assertTrue(retrogradeboard.is_valid())
        self.assertFalse(any(retrogradeboard.generate_pseudo_legal_unmoves()))
        self.assertEqual(set(retrogradeboard.pseudo_legal_unmoves), set(retrogradeboard.generate_pseudo_legal_unmoves()))
        self.assertFalse(any(retrogradeboard.mirror().generate_pseudo_legal_unmoves()))

    def test_pseudo_legal_uncapture(self):
        retrogradeboard = RetrogradeBoard(fen="3k4/8/8/8/4K3/8/8/8 b - - 0 1", pocket_b="NBRQP")
        self.assertTrue(retrogradeboard.is_valid())
        unmoves = []
        for p in "NBRQP ":
            if p == " ": #hack for moves which aren't uncapture
                p = ""
            unmoves += [f"{p}e4e5", f"{p}e4e3", f"{p}e4d3", f"{p}e4d4", f"{p}e4d5", f"{p}e4f3", f"{p}e4f4", f"{p}e4f5"]
        unmoves_set = set([UnMove.from_retro_uci(i) for i in unmoves])
        unmoves_set_b = set([UnMove.from_retro_uci(i).mirror() for i in unmoves])
        self.assertEqual(unmoves_set, set(retrogradeboard.generate_pseudo_legal_unmoves()))
        self.assertEqual(set(retrogradeboard.pseudo_legal_unmoves), set(retrogradeboard.generate_pseudo_legal_unmoves()))
        self.assertEqual(unmoves_set_b, set(retrogradeboard.mirror().generate_pseudo_legal_unmoves()))

    def test_pseudo_legal_uncapture_by_pawn(self):
        retrogradeboard = RetrogradeBoard(fen="2k5/8/8/8/5P2/8/nn6/Kn6 b - - 0 1", pocket_b="NBRQP")
        self.assertTrue(retrogradeboard.is_valid())
        unmoves = ["f4f3", "f4f2"]
        for p in "NBRQP":
            unmoves += [f"{p}f4g3", f"{p}f4e3"]
        unmoves_set = set([UnMove.from_retro_uci(i) for i in unmoves])
        unmoves_set_b = set([UnMove.from_retro_uci(i).mirror() for i in unmoves])
        self.assertEqual(unmoves_set, set(retrogradeboard.generate_pseudo_legal_unmoves()))
        self.assertEqual(set(retrogradeboard.pseudo_legal_unmoves), set(retrogradeboard.generate_pseudo_legal_unmoves()))
        self.assertEqual(unmoves_set_b, set(retrogradeboard.mirror().generate_pseudo_legal_unmoves()))

    def test_pseudo_legal_uncapture_by_pawn_illegal(self):
        """All Uncaptures are illegal here because the destination square is taken"""
        retrogradeboard = RetrogradeBoard(fen="2k5/8/8/8/5P2/4q1q1/nn6/Kn6 b - - 0 1", pocket_b="NBRQP")
        self.assertTrue(retrogradeboard.is_valid())
        unmoves = ["f4f3", "f4f2"]
        unmoves_set = set([UnMove.from_retro_uci(i) for i in unmoves])
        unmoves_set_b = set([UnMove.from_retro_uci(i).mirror() for i in unmoves])
        self.assertEqual(unmoves_set, set(retrogradeboard.generate_pseudo_legal_unmoves()))
        self.assertEqual(set(retrogradeboard.pseudo_legal_unmoves), set(retrogradeboard.generate_pseudo_legal_unmoves()))
        self.assertEqual(unmoves_set_b, set(retrogradeboard.mirror().generate_pseudo_legal_unmoves()))

    def test_pseudo_legal_uncapture_pawn_illegal(self):
        """Pawns can't be uncaptured on the first or the last rank"""
        retrogradeboard = RetrogradeBoard(fen="7k/8/8/8/8/8/8/7K b - - 0 1", pocket_b="NBRQP")
        self.assertTrue(retrogradeboard.is_valid())
        unmoves = []
        for p in "NBRQ ":
            if p == " ": #hack for moves which aren't uncapture
                p = ""
            unmoves += [f"{p}h1g2", f"{p}h1h2", f"{p}h1g1"]
        unmoves_set = set([UnMove.from_retro_uci(i) for i in unmoves])
        unmoves_set_b = set([UnMove.from_retro_uci(i).mirror() for i in unmoves])
        self.assertEqual(unmoves_set, set(retrogradeboard.generate_pseudo_legal_unmoves()))
        self.assertEqual(set(retrogradeboard.pseudo_legal_unmoves), set(retrogradeboard.generate_pseudo_legal_unmoves()))
        self.assertEqual(unmoves_set_b, set(retrogradeboard.mirror().generate_pseudo_legal_unmoves()))

    def test_pseudo_legal_unpromote(self):
        retrogradeboard = RetrogradeBoard(fen="1N5k/8/8/8/8/8/6nn/6nK b - - 0 1", pocket_w="1")
        self.assertTrue(retrogradeboard.is_valid())
        unmoves = ["b8a6", "b8c6", "b8d7", "Ub8b7"]
        unmoves_set = set([UnMove.from_retro_uci(i) for i in unmoves])
        unmoves_set_b = set([UnMove.from_retro_uci(i).mirror() for i in unmoves])
        self.assertEqual(unmoves_set, set(retrogradeboard.generate_pseudo_legal_unmoves()))
        self.assertEqual(set(retrogradeboard.pseudo_legal_unmoves), set(retrogradeboard.generate_pseudo_legal_unmoves()))
        self.assertEqual(unmoves_set_b, set(retrogradeboard.mirror().generate_pseudo_legal_unmoves()))

    def test_pseudo_legal_multiple_unpromotion(self):
        retrogradeboard = RetrogradeBoard(fen="N2N3k/1qq2q2/1qq1q3/8/8/8/6nn/6nK b - - 0 1", pocket_w="1")
        self.assertTrue(retrogradeboard.is_valid())
        unmoves = ["Ua8a7", "Ud8d7"]
        unmoves_set = set([UnMove.from_retro_uci(i) for i in unmoves])
        unmoves_set_b = set([UnMove.from_retro_uci(i).mirror() for i in unmoves])
        self.assertEqual(unmoves_set, set(retrogradeboard.generate_pseudo_legal_unmoves()))
        self.assertEqual(set(retrogradeboard.pseudo_legal_unmoves), set(retrogradeboard.generate_pseudo_legal_unmoves()))
        self.assertEqual(unmoves_set_b, set(retrogradeboard.mirror().generate_pseudo_legal_unmoves()))

    def test_pseudo_legal_unpromote_illegal(self):
        """There's a piece on the unpromotion square"""
        retrogradeboard = RetrogradeBoard(fen="1N5k/1q6/8/8/8/8/6nn/6nK b - - 0 1", pocket_w="1")
        self.assertTrue(retrogradeboard.is_valid())
        unmoves = ["b8a6", "b8c6", "b8d7"]
        unmoves_set = set([UnMove.from_retro_uci(i) for i in unmoves])
        unmoves_set_b = set([UnMove.from_retro_uci(i).mirror() for i in unmoves])
        self.assertEqual(unmoves_set, set(retrogradeboard.generate_pseudo_legal_unmoves()))
        self.assertEqual(set(retrogradeboard.pseudo_legal_unmoves), set(retrogradeboard.generate_pseudo_legal_unmoves()))
        self.assertEqual(unmoves_set_b, set(retrogradeboard.mirror().generate_pseudo_legal_unmoves()))

    def test_pseudo_legal_unpromote_and_uncapture(self):
        """Even if there's a pawn in opponent's pocket it can't be uncaptured on the first or the last rank"""
        retrogradeboard = RetrogradeBoard(fen="1N6/3q3k/q1q5/8/8/8/6nn/6nK b - - 0 1",pocket_w="1",pocket_b="NBRQP")
        self.assertTrue(retrogradeboard.is_valid())
        unmoves = ["Ub8b7"]
        for p in "NBRQ":
            unmoves += [f"U{p}b8a7", f"U{p}b8c7"]
        unmoves_set = set([UnMove.from_retro_uci(i) for i in unmoves])
        unmoves_set_b = set([UnMove.from_retro_uci(i).mirror() for i in unmoves])
        self.assertEqual(unmoves_set, set(retrogradeboard.generate_pseudo_legal_unmoves()))
        self.assertEqual(set(retrogradeboard.pseudo_legal_unmoves), set(retrogradeboard.generate_pseudo_legal_unmoves()))
        self.assertEqual(unmoves_set_b, set(retrogradeboard.mirror().generate_pseudo_legal_unmoves()))

    def test_pseudo_legal_unpromote_and_uncapture_on_A_file(self):
        retrogradeboard = RetrogradeBoard(fen="N7/2q4k/1q6/8/8/8/6nn/6nK b - - 0 1",pocket_w="1",pocket_b="NBRQP")
        self.assertTrue(retrogradeboard.is_valid())
        unmoves = ["Ua8a7"]
        for p in "NBRQ":
            unmoves.append(f"U{p}a8b7")
        unmoves_set = set([UnMove.from_retro_uci(i) for i in unmoves])
        unmoves_set_b = set([UnMove.from_retro_uci(i).mirror() for i in unmoves])
        self.assertEqual(unmoves_set, set(retrogradeboard.generate_pseudo_legal_unmoves()))
        self.assertEqual(set(retrogradeboard.pseudo_legal_unmoves), set(retrogradeboard.generate_pseudo_legal_unmoves()))
        self.assertEqual(unmoves_set_b, set(retrogradeboard.mirror().generate_pseudo_legal_unmoves()))

    def test_pseudo_legal_unpromote_and_uncapture_on_H_file(self):
        retrogradeboard = RetrogradeBoard(fen="7N/k4q2/6q1/8/8/8/nn6/Kn6 b - - 0 1",pocket_w="1",pocket_b="NBRQP")
        self.assertTrue(retrogradeboard.is_valid())
        unmoves = ["Uh8h7"]
        for p in "NBRQ":
            unmoves.append(f"U{p}h8g7")
        unmoves_set = set([UnMove.from_retro_uci(i) for i in unmoves])
        unmoves_set_b = set([UnMove.from_retro_uci(i).mirror() for i in unmoves])
        self.assertEqual(unmoves_set, set(retrogradeboard.generate_pseudo_legal_unmoves()))
        self.assertEqual(set(retrogradeboard.pseudo_legal_unmoves), set(retrogradeboard.generate_pseudo_legal_unmoves()))
        self.assertEqual(unmoves_set_b, set(retrogradeboard.mirror().generate_pseudo_legal_unmoves()))

    def test_final_pseudo_legal_unmoves(self):
        """
        Iterating 2 ply deep on every pseudo legal unmoves then retropoppin and see if they're still equal at the end.
        Choosing a fairly complicated position to test a bit everything: 1N6/1r5k/8/8/2P5/8/1Q2P3/n5Kb w - - 0 1
        Also used as a perft function, with the canonical result being the first one obtained without bug, which could be wrong too. 
        """
        retrogradeboard = RetrogradeBoard(fen="1N6/1r5k/8/8/2P5/8/1Q2P3/n5Kb w - - 0 1",pocket_w="2PNBRQ",pocket_b="3NBRQP")
        retrogradeboard_2 = RetrogradeBoard(fen="1N6/1r5k/8/8/2P5/8/1Q2P3/n5Kb w - - 0 1",pocket_w="2PNBRQ",pocket_b="3NBRQP")
        self.assertTrue(retrogradeboard.is_valid())
        unmove_counter = 0
        for unmove in retrogradeboard.pseudo_legal_unmoves:
            unmove_counter += 1
            retrogradeboard.retropush(unmove)
            for unmove_2 in retrogradeboard.pseudo_legal_unmoves:
                unmove_counter += 1
                retrogradeboard.retropush(unmove_2)
                retrogradeboard.retropop()

            retrogradeboard.retropop()

        self.assertEqual(retrogradeboard, retrogradeboard_2)
        self.assertEqual(unmove_counter, 22952)

    def test_attacks_mask_of_piece_at_square(self):
        retrogradeboard = RetrogradeBoard(fen="7r/2Q3p1/8/2k5/3N4/2B1N3/3K4/8 b - - 0 1")
        self.assertTrue(retrogradeboard.is_valid())
        self.assertRaises(ValueError, retrogradeboard.attacks_mask_of_piece_at_square, chess.G7, chess.E4)
        dic = {(chess.D2, chess.A1): [chess.A2, chess.B1, chess.B2], # King
               (chess.D2, chess.C5): [chess.B4, chess.B5, chess.B6,chess.C4, chess.C6, chess.D4, chess.D5, chess.D6], # King
               (chess.H8, chess.H8): (chess.BB_RANK_8 |chess.BB_FILE_H) & ~chess.BB_SQUARES[chess.H8], # Rook
               (chess.H8, chess.D3): [chess.D2, chess.D4, chess.C3, chess.E3], # Rook
               (chess.C7, chess.A8): (chess.BB_RANK_8 |chess.BB_FILE_A) & ~chess.BB_SQUARES[chess.A8] | chess.BB_SQUARES[chess.B7] | chess.BB_SQUARES[chess.C6] | chess.BB_SQUARES[chess.D5] | chess.BB_SQUARES[chess.E4] | chess.BB_SQUARES[chess.F3] | chess.BB_SQUARES[chess.G2] | chess.BB_SQUARES[chess.H1], # Queen
               (chess.C3, chess.A1): [chess.B2, chess.C3, chess.D4], # Bishop
        }
        for t, l in dic.items():
            with self.subTest(coord=t):
                self.assertEqual(retrogradeboard.attacks_mask_of_piece_at_square(*t),chess.SquareSet(l))


    def test_conditions_for_legality_of_move(self):
        retrogradeboard = RetrogradeBoard(fen="1k5R/8/Kn6/nn5p/8/8/8/8 b - - 0 1")
        #Actual condition used in `legal_unmoves`
        self.assertFalse(not retrogradeboard.attacks_mask_of_piece_at_square(chess.H8, chess.C8) & retrogradeboard.occupied_co[not retrogradeboard.retro_turn] & retrogradeboard.kings)


    ####################
    # Helper functions #
    ####################

    def check_pos_unmoves(self, fen: str, unmoves: list, pocket_b: str = "", pocket_w: str = "", debug: bool = False):
        """Check if the rboard with the given `fen` has only the `unmoves` legals."""
        retrogradeboard = RetrogradeBoard(fen=fen,pocket_b=pocket_b,pocket_w=pocket_w)
        if debug:
            retrogradeboard.pp()
            king_mask = retrogradeboard.kings & retrogradeboard.occupied_co[not retrogradeboard.retro_turn]
            king = chess.msb(king_mask)
            print(chess.SquareSet(retrogradeboard.retro_slider_blockers(king)))
        self.assertTrue(retrogradeboard.is_valid())
        unmoves_set = set([UnMove.from_retro_uci(i) for i in unmoves])
        unmoves_set_b = set([UnMove.from_retro_uci(i).mirror() for i in unmoves])
        if debug:
            print(retrogradeboard.pseudo_legal_unmoves)
            print(retrogradeboard.legal_unmoves)
        self.assertEqual(unmoves_set, set(retrogradeboard.generate_legal_unmoves()))
        self.assertEqual(set(retrogradeboard.legal_unmoves), set(retrogradeboard.generate_legal_unmoves()))
        self.check_legal_unmoves(retrogradeboard, debug)
        if debug:
            retrogradeboard.mirror().pp()
            print(retrogradeboard.mirror().legal_unmoves)
            print(retrogradeboard.mirror().mirror().legal_unmoves)
        self.assertEqual(unmoves_set_b, set(retrogradeboard.mirror().generate_legal_unmoves()))
        self.assertEqual(retrogradeboard, retrogradeboard.mirror().mirror())

    def check_legal_unmoves(self, rboard: RetrogradeBoard, debug: bool = False) -> int:
        """
        Checks that pos is legal after every unmove and that the move corresponding to the unmove is legal too
        Returns the number of legal unmoves from the position"""
        counter = 0
        if debug:
            print("Start validing move")
        for unmove in rboard.legal_unmoves:
            with self.subTest(rboard=rboard.fen(), unmove=unmove):
                counter += 1
                piece_type_moved = rboard.piece_type_at(unmove.from_square)
                rboard.retropush(unmove)
                if debug:
                    print(f"after {unmove}")
                    rboard.pp()
                    print(rboard.fen())
                self.assertTrue(rboard.is_valid())
                move_corresponding_to_unmove = chess.Move(from_square=unmove.to_square,to_square=unmove.from_square,
                                                        promotion=piece_type_moved if unmove.unpromotion else None)
                self.assertTrue(rboard.is_legal(move_corresponding_to_unmove))
                rboard.retropop()
                if debug:
                    print("after retropoping")
                    rboard.pp()
        return counter

    #################

    def test_legal_unmoves_move_checker_away(self):
        """Any move resulting in checking opponent's king is retro-illegal"""
        self.check_pos_unmoves("1k5R/8/Kn6/nn5p/8/8/8/8 b - - 0 1", ["h8h7", "h8h6"])

    def test_legal_unmoves_blocker_of_check(self):
        """The only unmoves to find here is how to block the checker"""
        self.check_pos_unmoves("1k5R/7p/1K3N2/8/8/8/8/8 b - - 0 1", ["f6e8", "f6g8"])

    def test_legal_unmoves_pin(self):
        """The knight is pinned"""
        self.check_pos_unmoves("3k1N1R/8/7p/8/8/8/8/K7 b - - 0 1", ["h8g8", "h8h7", "a1b1", "a1b2", "a1a2"])

    def test_legal_unmoves_knight_cant_be_blocked(self):
        self.check_pos_unmoves("3kn3/8/3K4/8/8/8/8/q7 w - - 0 1", ["e8c7", "e8f6", "e8f6", "e8g7"])

    def test_legal_unmoves_pawns_cant_be_blocked(self):
        self.check_pos_unmoves("3k4/8/8/4p3/3K4/8/8/1q6 w - - 0 1", ["e5e6", "e5e7"])

    def test_legal_unmoves_checkmate_discarded(self):
        """Any move resulting in checkmating (bc checkmate => check) opponent's king is retro-illegal"""
        self.check_pos_unmoves("k7/1Q6/1Kb5/8/8/8/8/8 b - - 0 1", ["b7c7", "b7d7", "b7e7", "b7f7", "b7g7", "b7h7"])

    def test_legal_unmoves_check_discarded_2(self):
        """Any move resulting in checking opponent's king is retro-illegal.
        This test if unmoves like f8h8 are correclty labelled as illegal (on static eval rook could be blocking the attack mask) 
        """
        self.check_pos_unmoves("1k3R2/8/Kn6/nn3p2/8/8/8/8 b - - 0 1", ["f8f7", "f8f6"])

    def test_legal_unmoves_uncapture_blocking_check(self):
        """Any move resulting in checking opponent's king is retro-illegal.
        This test if unmoves like Qf8h8 are correclty labelled as legal, since the uncaptured piece is blocking the attack) 
        """
        unmoves = ["f8f7", "f8f6"]
        for p in "NBRQ":
            unmoves += [f"{p}f8f7", f"{p}f8f6", f"{p}f8g8", f"{p}f8h8"]
        self.check_pos_unmoves("1k3R2/8/Kn6/nn3p2/8/8/8/8 b - - 0 1", unmoves, pocket_b="PNBRQ")

    def test_legal_pawn_unmove_uncapture(self):
        unmoves = ["g3g2"]
        for p in "NBRQP":
            unmoves += [f"{p}g3f2", f"{p}g3h2"]
        self.check_pos_unmoves("8/8/8/8/5k2/6P1/8/1K6 b - - 0 1", unmoves, pocket_b="PNBRQ")

    def test_legal_pawn_unpromotion(self):
        """Promotion is illegal here because it would put king in check"""
        self.check_pos_unmoves("3kR3/8/8/8/8/8/8/3K4 b - - 0 1", ["e8e7", "e8e6", "e8e5", "e8e4", "e8e3", "e8e2", "e8e1"], pocket_w="1")

    def test_legal_pawn_unpromotion_uncapture(self):
        unmoves = []
        for p in "NBRQ ":
            if p == " ":
                p = ""
            unmoves += [f"{p}e8e7", f"{p}e8e6", f"{p}e8e5", f"{p}e8e4", f"{p}e8e3", f"{p}e8e2", f"{p}e8e1"]
            if p != "":
                unmoves += [f"U{p}e8d7", f"U{p}e8f7",
                            f"{p}e8f8", f"{p}e8g8", f"{p}e8h8"]
        self.check_pos_unmoves("3kR3/8/8/8/8/8/8/3K4 b - - 0 1", unmoves, pocket_w="1", pocket_b="NBRQ")

    def test_legal_double_check_possible_rook_bishop(self):
        """When in double check there's only one deplacement possible, which means only one unmove if no uncaptures are possible"""
        self.check_pos_unmoves("3k4/8/8/3R4/7B/8/8/4K3 b - - 0 1", ["d5g5"])

    def test_legal_double_check_possible_rook_bishop_with_uncapture(self):
        """When in double check there's only one deplacement possible, which means only one unmove if no uncaptures are possible"""
        unmoves = ["d5g5"]
        for p in "NBRQP":
            unmoves.append(f"{p}d5g5")
        self.check_pos_unmoves("3k4/8/8/3R4/7B/8/8/4K3 b - - 0 1", unmoves, pocket_b="NBRQP")

    def test_legal_double_check_impossible_rook_bishop(self):
        """When in double check there's only one deplacement possible, which means only one unmove if no uncaptures are possible"""
        self.check_pos_unmoves("8/8/3R1k2/8/7B/8/8/4K3 b - - 0 1", [])

    def test_legal_double_check_queen_knight(self):
        """When in double check there's only one deplacement possible, which means only one unmove if no uncaptures are possible"""
        self.check_pos_unmoves("8/4k3/2N5/8/8/4Q3/8/4K3 b - - 0 1", ["c6e5"])

    def test_legal_double_check_queen_knight_impossible(self):
        self.check_pos_unmoves("4k3/2N5/4Q3/8/8/8/8/3K4 b - - 0 1", [])

    def test_legal_double_check_pawns(self):
        """There can't be legal unmove when two pawns are checking"""
        self.check_pos_unmoves("4k3/3P1P2/8/8/8/8/8/3K4 b - - 0 1", [])

    def test_legal_double_check_knights(self):
        """There can't be legal unmove when two knights are checking"""
        self.check_pos_unmoves("4k3/2N5/5N2/8/8/8/8/3K4 b - - 0 1", [])

    def test_legal_double_check_knight_pawn(self):
        """There can't be legal unmove when two knights are checking"""
        self.check_pos_unmoves("4k3/2N2P2/8/8/8/8/8/3K4 b - - 0 1", [])

    def test_legal_double_queens_impossible(self):
        """There can't be legal unmove when two queens are checking and *no uncapture + unpromotion*"""
        self.check_pos_unmoves("4kQ2/8/4Q3/8/8/8/8/3K4 b - - 0 1", [])

    def test_legal_double_queens(self):
        """There's only one legal unmove when two queens are checking *with uncapture unpromotion*"""
        unmoves = []
        for p in "NBRQ":
            unmoves.append(f"U{p}f8e7")
        self.check_pos_unmoves("4kQ2/8/4Q3/8/8/8/8/3K4 b - - 0 1", unmoves, pocket_w="1", pocket_b="PNBRQ")

    def test_legal_triple_check(self):
        self.check_pos_unmoves("8/1R1k2R1/8/8/8/3Q4/8/3K4 b - - 0 1", [], pocket_w="1PNQRB", pocket_b="PNBRQ")

    def test_final_legal_unmoves(self):
        """
        Iterating 2 ply deep on every legal unmoves then retropopping and see if they're still equal at the end.
        Choosing a fairly complicated position to test a bit everything: q4N2/1p5k/8/8/6P1/4Q3/1K1PB3/7r b - - 0 1
        Also used as a perft function, with the canonical result being the first one obtained without bug, which could be wrong too. 
        """
        retrogradeboard = RetrogradeBoard(fen="q4N2/1p5k/8/8/6P1/4Q3/1K1PB3/7r b - - 0 1",pocket_w="2PNBRQ",pocket_b="3NBRQP")
        retrogradeboard_2 = RetrogradeBoard(fen="q4N2/1p5k/8/8/6P1/4Q3/1K1PB3/7r b - - 0 1",pocket_w="2PNBRQ",pocket_b="3NBRQP")
        self.assertTrue(retrogradeboard.is_valid())
        unmove_counter = 0
        for unmove in retrogradeboard.legal_unmoves:
            unmove_counter += 1
            piece_type_moved = retrogradeboard.piece_type_at(unmove.from_square)
            retrogradeboard.retropush(unmove)
            self.assertTrue(retrogradeboard.is_valid())
            move_corresponding_to_unmove = chess.Move(from_square=unmove.to_square,to_square=unmove.from_square,
                                                    promotion=piece_type_moved if unmove.unpromotion else None)
            self.assertTrue(retrogradeboard.is_legal(move_corresponding_to_unmove))
            # Iterating one ply deeper
            unmove_counter += self.check_legal_unmoves(retrogradeboard)
            retrogradeboard.retropop()

        self.assertEqual(retrogradeboard, retrogradeboard_2)
        self.assertEqual(unmove_counter, 3975)

######
#Main#
######

if __name__ == "__main__":
    print("#"*80)
    unittest.main()