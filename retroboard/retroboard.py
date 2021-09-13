#!/usr/local/bin/python3
#coding: utf-8

"""
A chess retrograde move generator
"""

from __future__ import annotations

import chess
import copy
import re

from chess import BB_ALL, BB_EMPTY, BB_SQUARES, BLACK, scan_reversed, Square, WHITE
from dataclasses import dataclass

from typing import Dict, Optional, List, Iterator, Iterable, Tuple, Hashable

###########
#Constants#
###########

RETRO_UCI_REGEX = re.compile(
    r"(?P<unpromotion>U?)"
    r"(?P<uncapture>[PNBRQ]?)"
    r"(?P<uci_part>.*)"
    )

#########
#Classes#
#########

@dataclass
class UnMove(chess.Move):
    """A move made on a `RetrogradeBoard`
    -A piece move from a square to another (pawns move backward)
    -A piece uncapture a piece, leaving it at the source square
    -A piece unpromote
    Castling and en-passant are NOT supported
    """
    uncapture: Optional[chess.PieceType] = None
    unpromotion: bool = False

    def to_tuple(self) -> Tuple[Square, Square, Optional[chess.PieceType], bool]:
        return (self.from_square, self.to_square, self.uncapture, self.unpromotion)

    def __hash__(self) -> int:
        return hash(self.to_tuple())

    def __eq__(self, unmove: object) -> bool:
            if isinstance(unmove, UnMove):
                return self.to_tuple() == unmove.to_tuple()
            return NotImplemented

    def __repr__(self) -> str:
        return self.retro_uci()

    def retro_uci(self) -> str:
        """
        Gets a retro_UCI string for the move.

        For example, a move from a7 to a8 would be ``a7a8``
        See `from_retro_uci` doc for more explanation of specific syntax of retro_uci over uci.
        """
        res = ""
        if self.unpromotion:
            res += "U"
        if self.uncapture:
            res += chess.piece_symbol(self.uncapture).upper()
        return res + super().uci()

    def mirror(self) -> UnMove:
        return type(self)(from_square=chess.square_mirror(self.from_square), 
                   to_square=chess.square_mirror(self.to_square),
                   uncapture=self.uncapture,
                   unpromotion=self.unpromotion)

    @classmethod
    def from_retro_uci(cls, retro_uci: str) -> UnMove:
        """
        movements are represented with uci, but for uncapture and unpromote
        a special syntax is used:
        -Uncapture: the piece left at the source square is indicated at the beginning, follow by normal uci move.
        e.g: "Re2e4" the piece on e2 goes on e4 and leaves a Rook from the opposite color on e2. 
        -Unpromotion: "U" and after the square from which the piece will underpromote and the
        source square must be on the 8th or 1st rank, and dest square must be on first or second rank. 
        e.g: "Ue8e7". 
        An unpromotion can also be an uncapture, in this case it's noted "<PieceType>U<from_square><to_square>"
        e.g "UNe8e7"

        regex: r"U?[NBRQ]?([abcdefgh][1-8]){2}"

        Note: A unmove being accepted does not means it is for sure legal, just syntaxically correct
        """
        match = RETRO_UCI_REGEX.fullmatch(retro_uci)
        if not match:
            raise ValueError(f"retro_uci not matching regex: {retro_uci}")

        unpromotion = match.group("unpromotion") != ""
        if match.group("uncapture"):
            uncapture: Optional[chess.PieceType] = chess.Piece.from_symbol(match.group("uncapture")).piece_type
        else:
            uncapture = None
        uci_part = match.group("uci_part")

        try:
            move = chess.Move.from_uci(uci_part)
        except ValueError as e:
            raise ValueError(f"Invalid retro_uci: {retro_uci}")
        return cls(from_square=move.from_square, 
                   to_square=move.to_square,
                   uncapture=uncapture,
                   unpromotion=unpromotion)

class NotInPocket(Exception):
    def __init__(self, piece_type: chess.PieceType) -> None:
        super().__init__(f"Trying to remove a piece ({piece_type}) which is not in the pocket")

class TooManyNumberForPocket(Exception):
    def __init__(self, symbols: Iterable[str]) -> None:
        super().__init__(f"More than one number given to generate a pocket: {symbols}") 

class RetrogradeBoardPocket:
    """
    A RetrogradeBoard pocket with a counter for each piece type. 
    It stores the pieces than can be uncaptured by each color.
    
    `self.nb_unpromotion` is the number of pieces than can unpromote into a pawn.
    By default it is set to 0
    """

    def __init__(self, symbols: Iterable[str] = "") -> None:
        """
        `symbols` is a list of `PIECE_SYMBOLS` with one number, corresponding 
        to `self.nb_unpromotion`. If no number is given, then it is set to 0.
        If more than one number is given, raise `BadPocketInit`
        """
        self.nb_unpromotion = 0
        self.pieces: Dict[chess.PieceType, int] = {}
        for symbol in symbols:
            try:
                nb_unpromotion = int(symbol)
                if self.nb_unpromotion:
                    raise TooManyNumberForPocket(symbols)
                self.nb_unpromotion = nb_unpromotion
            except ValueError:
                self.add(chess.PIECE_SYMBOLS.index(symbol.lower()))

    def __eq__(self, pocket: object) -> bool:
        if isinstance(pocket, RetrogradeBoardPocket):
            return str(self) == str(pocket)
        return NotImplemented

    def add(self, piece_type: chess.PieceType) -> None:
        """Adds a piece of the given type to this pocket."""
        self.pieces[piece_type] = self.pieces.get(piece_type, 0) + 1

    def remove(self, piece_type: chess.PieceType) -> None:
        """Removes a piece of the given type from this pocket."""
        try:
            self.pieces[piece_type] -= 1
        except KeyError:
            raise NotInPocket(piece_type)

    def get_nb_unpromotion(self) -> int:
        return max(self.nb_unpromotion, 0)

    def decrement_unpromotion(self) -> None:
        self.nb_unpromotion -= 1

    def count(self, piece_type: chess.PieceType) -> int:
        """Returns the number of pieces of the given type in the pocket."""
        return self.pieces.get(piece_type, 0)

    def reset(self) -> None:
        """Clears the pocket."""
        self.pieces.clear()

    def __str__(self) -> str:
        return "".join(chess.piece_symbol(pt) * self.count(pt) for pt in reversed(chess.PIECE_TYPES)) + str(self.nb_unpromotion)

    def __len__(self) -> int:
        return sum(self.pieces.values())

    def __repr__(self) -> str:
        return f"RetrogradeBoardPocket('{self}, {self.nb_unpromotion}')"

    def copy(self) -> RetrogradeBoardPocket:
        """Returns a copy of this pocket."""
        pocket = type(self)()
        pocket.pieces = copy.copy(self.pieces)
        pocket.nb_unpromotion = self.nb_unpromotion
        return pocket


class PseudoLegalUnMoveGenerator:

    def __init__(self, rboard: RetrogradeBoard) -> None:
        self.rboard = rboard

    def __bool__(self) -> bool:
        return any(self.rboard.generate_pseudo_legal_unmoves())

    def count(self) -> int:
        # List conversion is faster than iterating.
        return len(list(self))

    def __iter__(self) -> Iterator[UnMove]:
        return self.rboard.generate_pseudo_legal_unmoves()

    def __contains__(self, unmove: UnMove) -> bool:
        return self.rboard.is_pseudo_retrolegal(unmove)

    def __repr__(self) -> str:
        builder = []

        for move in self:
            builder.append(move.retro_uci())

        retro_uci = ", ".join(builder)
        return f"<PseudoLegalUnMoveGenerator at {id(self):#x} ({retro_uci})>"

class LegalUnMoveGenerator:

    def __init__(self, rboard: RetrogradeBoard) -> None:
        self.rboard = rboard

    def __bool__(self) -> bool:
        return any(self.rboard.generate_legal_unmoves())

    def count(self) -> int:
        # List conversion is faster than iterating.
        return len(list(self))

    def __iter__(self) -> Iterator[UnMove]:
        return self.rboard.generate_legal_unmoves()

    def __contains__(self, unmove: UnMove) -> bool:
        return self.rboard.is_retrolegal(unmove)

    def __repr__(self) -> str:
        builder = []

        for unmove in self:
            builder.append(unmove.retro_uci())

        retro_uci = ", ".join(builder)
        return f"<LegalUnMoveGenerator at {id(self):#x} ({retro_uci})>"


class RetrogradeBoard(chess.Board):
    """
    A `Board` where `Unmove` are played and all legal `Unmove` can be generated, on top of every thing a `Board` can do.
    At every time the position must be legal, which force us to swap the turn because 
    when it's black to play, it means that white made the last move (and we are generating those)
    `halfmove_clock` is not updated
    """
    retro_turn: chess.Color
    unmove_stack: List[UnMove] = []
    _retrostack: List[_RetrogradeBoardState] = []
    pockets: Tuple[RetrogradeBoardPocket, RetrogradeBoardPocket]

    def __init__(self, 
                fen: Optional[str] = chess.STARTING_FEN, 
                pocket_w: str = "", 
                pocket_b: str = "") -> None:
        super().__init__(fen)
        self.pockets = (RetrogradeBoardPocket(pocket_b), RetrogradeBoardPocket(pocket_w))
        self.retro_turn = not self.turn

    def _transposition_key(self) -> Hashable:
        return (super()._transposition_key(),
                self.retro_turn,
                str(self.pockets[BLACK]), str(self.pockets[WHITE]))

    def is_valid(self) -> bool:
        """
        Checks some basic validity requirements.

        See :func:`~chess.Board.status()` for details.
        """
        status = self.status()
        return status == chess.STATUS_VALID or status == chess.STATUS_TOO_MANY_CHECKERS

    def swap_turns(self) -> None:
        self.turn = not self.turn
        self.retro_turn = not self.retro_turn

    def mirror(self) -> RetrogradeBoard:
        rboard = super().mirror()
        rboard.retro_turn = not self.retro_turn
        rboard.pockets = (self.pockets[WHITE].copy(), self.pockets[BLACK].copy())
        return rboard

    def retropop(self) -> UnMove:
        """
        Restores the previous position and returns the last move from the stack.
        :raises: :exc:`IndexError` if the move stack is empty.
        """
        unmove = self.unmove_stack.pop()
        self._retrostack.pop().restore(self)
        return unmove

    def is_retro_zeroing(self, unmove: UnMove) -> bool:
        return bool(unmove.uncapture) or unmove.unpromotion or self.piece_type_at(unmove.from_square) == chess.PAWN

    def _board_state(self) -> _RetrogradeBoardState:
        return _RetrogradeBoardState(self)

    def retropush(self, unmove: UnMove) -> None:
        """
        Play a `UnMove` on the board.
        Castling and en-passant are NOT supported
        `halfmove_clock` is not updated

        Warning: Does not check for legality!
        """
        # Push unmove and remember board state.
        board_state = self._board_state()
        self.unmove_stack.append(unmove)
        self._retrostack.append(board_state)
        if self.retro_turn == BLACK:
            self.fullmove_number += 1
        # On a null move, simply swap turns and reset the en passant square.
        if not unmove:
            self.swap_turns()
            return

        from_bb = BB_SQUARES[unmove.from_square]
        to_bb = BB_SQUARES[unmove.to_square]
        piece_type = self._remove_piece_at(unmove.from_square)
        assert piece_type is not None, f"retropush() expects unmove to be retropseudo-legal, but got {unmove} in {self.board_fen()}"
        capture_square = unmove.to_square
        captured_piece_type = self.piece_type_at(capture_square)
        assert captured_piece_type is None, f"retropush() expects unmove to be pseudo-retrolegal, but got {unmove} in {self.board_fen()} with {captured_piece_type}"
        # Put the piece on the target square, or a pawn if it's unpromoting
        if unmove.unpromotion:
            self._set_piece_at(unmove.to_square, chess.PAWN, self.retro_turn)
            self.pockets[self.retro_turn].decrement_unpromotion()
        else:
            self._set_piece_at(unmove.to_square, piece_type, self.retro_turn)
        # Uncaptures.
        if unmove.uncapture:
            self._set_piece_at(unmove.from_square, unmove.uncapture, not self.retro_turn)
            self.pockets[not self.retro_turn].remove(unmove.uncapture)
        # Swap turn.
        self.swap_turns()

    @property
    def pseudo_legal_unmoves(self) -> PseudoLegalUnMoveGenerator:
        return PseudoLegalUnMoveGenerator(self)

    def generate_pseudo_legal_unmoves(self, from_mask: chess.Bitboard = BB_ALL, to_mask: chess.Bitboard = BB_ALL) -> Iterator[UnMove]:
        our_pieces = self.occupied_co[self.retro_turn]

        # Generate piece unmoves.
        non_pawns = our_pieces & ~self.pawns & from_mask
        for from_square in scan_reversed(non_pawns):
            unmoves_mask = self.attacks_mask(from_square) & ~self.occupied & to_mask
            for to_square in scan_reversed(unmoves_mask):
                yield UnMove(from_square, to_square)
                # Generate uncaptures
                yield from self.generate_pseudo_legal_uncaptures(from_square, to_square)

            # Unpromotion
            # If the piece is not on the backrank (7 for white, 0 for black) or can't unpromote, pass
            if not (chess.square_rank(from_square) == 7*self.retro_turn and 
                self.pockets[self.retro_turn].get_nb_unpromotion()):
                continue

            to_square = from_square - 8 if self.retro_turn else from_square + 8
            if  BB_SQUARES[to_square] & ~self.occupied:
                    yield UnMove(from_square, to_square,unpromotion=True)

            # Uncapture and unpromotion
            to_square_uncaptures = []
            if chess.square_file(from_square) != 0: # A file
                to_square_uncaptures.append(to_square - 1)
            if chess.square_file(from_square) != 7: # H file
                to_square_uncaptures.append(to_square + 1)

            for to_square_uncapture in to_square_uncaptures:
                if  BB_SQUARES[to_square_uncapture] & ~self.occupied:
                        yield from self.generate_pseudo_legal_uncaptures(from_square, to_square_uncapture,unpromotion=True)

        # The remaining unmoves are all pawn ones.
        pawns = self.pawns & self.occupied_co[self.retro_turn] & from_mask
        if not pawns:
            return

        # Generate pawn uncaptures.
        for from_square in scan_reversed(pawns):
            targets = (
                chess.BB_PAWN_ATTACKS[not self.retro_turn][from_square] & ~self.occupied & to_mask)

            for to_square in scan_reversed(targets):
                if not chess.square_rank(to_square) in [0, 7]:
                    yield from self.generate_pseudo_legal_uncaptures(from_square, to_square)

        # Prepare pawn advance generation.
        if self.retro_turn == BLACK:
            single_moves = pawns << 8 & ~self.occupied
            double_moves = single_moves << 8 & ~self.occupied & chess.BB_RANK_7
        else:
            single_moves = pawns >> 8 & ~self.occupied
            double_moves = single_moves >> 8 & ~self.occupied & chess.BB_RANK_2

        single_moves &= to_mask
        double_moves &= to_mask

        # Generate single pawn unmoves.
        for to_square in scan_reversed(single_moves):
            from_square = to_square + (8 if self.retro_turn == WHITE else -8)

            if not chess.square_rank(to_square) in [0, 7]:
                yield UnMove(from_square, to_square)

        # Generate double pawn unmoves.
        for to_square in scan_reversed(double_moves):
            from_square = to_square + (16 if self.retro_turn == WHITE else -16)
            yield UnMove(from_square, to_square)

    def generate_pseudo_legal_uncaptures(self, from_square: Square, to_square: Square, unpromotion: bool = False) -> Iterator[UnMove]:
        for pt, count in self.pockets[not self.retro_turn].pieces.items():
                if count and (pt != chess.PAWN or not chess.BB_BACKRANKS & chess.BB_SQUARES[from_square]):
                    yield UnMove(from_square, to_square, uncapture=pt, unpromotion=unpromotion)

    def is_pseudo_retrolegal(self, unmove: UnMove) -> bool:
        return unmove in self.generate_pseudo_legal_unmoves()

    @property
    def legal_unmoves(self) -> LegalUnMoveGenerator:
        return LegalUnMoveGenerator(self)

    def attacks_mask_of_piece_at_square(self, from_square: Square, to_square: Square, uncapture: bool = False) -> chess.Bitboard:
        """
        Give the attack mask of the given piece type located at `from_square` at the `to_square`
        if `uncapture` is set to `False` then the mask go through the `from_square` (like if it's empty). Otherwise this square is considered occuped

        `chess.PAWN` type is not accepted
        """
        piece = self.piece_type_at(from_square)
        if uncapture:
            bb_pieces = self.occupied
        else:
            bb_pieces = self.occupied & ~BB_SQUARES[from_square]
        if piece == chess.PAWN:
            raise ValueError("PieceType PAWN is not accepted")

        elif piece == chess.KNIGHT:
            return chess.BB_KNIGHT_ATTACKS[to_square]
        elif piece == chess.KING:
            return chess.BB_KING_ATTACKS[to_square]
        else:
            attacks = 0
            if piece == chess.BISHOP or piece == chess.QUEEN:
                attacks = chess.BB_DIAG_ATTACKS[to_square][chess.BB_DIAG_MASKS[to_square] & bb_pieces]
            if piece == chess.ROOK or piece == chess.QUEEN:
                attacks |= (chess.BB_RANK_ATTACKS[to_square][chess.BB_RANK_MASKS[to_square] & bb_pieces] |
                            chess.BB_FILE_ATTACKS[to_square][chess.BB_FILE_MASKS[to_square] & bb_pieces])
            return attacks

    def generate_legal_unmoves(self, from_mask: chess.Bitboard = BB_ALL, to_mask: chess.Bitboard = BB_ALL) -> Iterator[UnMove]:
        """
        Generate legal unmoves, which are all the pseudo legal unmoves which do not put the opponent's king in check. 
        If the opponent's king is in check at the beginning of our turn, the only legal unmoves are those which stop it from being in check.
        """
        king_mask = self.kings & self.occupied_co[not self.retro_turn]
        king = chess.msb(king_mask)
        blockers = self.retro_slider_blockers(king)
        checkers = self.attackers_mask(self.retro_turn, king)
        nb_checkers = len(list(scan_reversed(checkers)))
        if nb_checkers > 2: # When there are 3 or more checkers there is no legal unmoves
            return

        if nb_checkers == 2:
            # Too many corner cases here, we need to push the move to see if the king is still in check.
            for unmove in self.generate_pseudo_legal_unmoves(from_mask, to_mask):
                if self.retro_is_safe(king_mask, blockers, unmove, checkers): # Is is worth checking it since we go bruteforce afterwards? Probably still saving some times
                    self.retropush(unmove)
                    still_check_after_unmove = self.attackers_mask(not self.retro_turn, king) #retro_turn has swapped
                    self.retropop()
                    if not still_check_after_unmove:
                        # There is at most only one "deplacement" in such position, but it means multiple unmoves with uncapture
                        yield unmove 
        else: #One checker
            for unmove in self.generate_pseudo_legal_unmoves(from_mask, to_mask):
                if self.retro_is_safe(king_mask, blockers, unmove, checkers):
                    yield unmove

    def retro_slider_blockers(self, king: Square) -> chess.Bitboard:
        rooks_and_queens = self.rooks | self.queens
        bishops_and_queens = self.bishops | self.queens
        snipers = ((chess.BB_RANK_ATTACKS[king][0] & rooks_and_queens) |
                   (chess.BB_FILE_ATTACKS[king][0] & rooks_and_queens) |
                   (chess.BB_DIAG_ATTACKS[king][0] & bishops_and_queens))
        blockers = 0

        for sniper in scan_reversed(snipers & self.occupied_co[self.retro_turn]):
            b = chess.between(king, sniper) & self.occupied
            # Add to blockers if exactly one piece in-between.
            if b and BB_SQUARES[chess.msb(b)] == b:
                blockers |= b

        return blockers & self.occupied_co[self.retro_turn]

    def retro_is_safe(self, king_mask: chess.Bitboard, blockers: chess.Bitboard, unmove: UnMove, checkers: chess.Bitboard = BB_EMPTY) -> bool:
        """
        Check if an unmove does not give check.
        Only work if there's at most one checker.
        """

        # If the unmove is a unpromotion but gives check, return False
        pawn_unmove = bool(unmove.unpromotion or self.piece_type_at(unmove.from_square) == chess.PAWN)
        if (pawn_unmove and
            chess.BB_PAWN_ATTACKS[self.retro_turn][unmove.to_square] & king_mask):  #type: ignore
            return False

        # If the unmove gives check, return False
        if not pawn_unmove and self.attacks_mask_of_piece_at_square(unmove.from_square, unmove.to_square, unmove.uncapture) & king_mask: #type: ignore
            return False

        # If we remove a blocker we'll put the king in check, so the unmove is invalid
        if not bool(not blockers & BB_SQUARES[unmove.from_square] or 
                        chess.ray(unmove.from_square, unmove.to_square) & king_mask
                        ):
            return False

        # No checker we can stop here
        # Put after the blocker check to avoid 'discovered attack'
        if not checkers:
            return True

        # If we move the checker and we know the unmove does not make check it means we move it to a square where it does not attack the king anymore
        # If the checker is a knight or a pawn and we won't move it the unmove can't be retrolegal
        # Otherwise the unmove need to put a piece between the checker (which is a slider) and the king
        return bool(checkers & BB_SQUARES[unmove.from_square] or
                    (not checkers & (self.pawns | self.knights) and chess.between(chess.msb(checkers), chess.msb(king_mask)) & BB_SQUARES[unmove.to_square])
                    )

    def is_retrolegal(self, unmove: UnMove) -> bool:
        return unmove in self.generate_legal_unmoves()

    def pp(self) -> None:
        print("\n" + 
            super().unicode(empty_square= ".")
            )

class _RetrogradeBoardState(chess._BoardState[RetrogradeBoard]):
    def __init__(self, rboard: RetrogradeBoard) -> None:
        super().__init__(rboard)
        self.pockets_w = rboard.pockets[WHITE].copy()
        self.pockets_b = rboard.pockets[BLACK].copy()
        self.retro_turn = rboard.retro_turn

    def restore(self, rboard: RetrogradeBoard) -> None:
        super().restore(rboard)
        rboard.pockets = (self.pockets_b.copy(), self.pockets_w.copy())
        rboard.retro_turn = self.retro_turn

######
#Main#
######


if __name__ == "__main__":
    un = UnMove.from_retro_uci("Ke2e4")
