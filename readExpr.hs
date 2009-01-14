{-# OPTIONS -fglasgow-exts #-}
--
-- experimenta:
--     read "(((1+2)*((5+34)-324))+(542*(425/2345)))"::Exp Ops
--

module ReadExpr where

import Char

data Exp o = Const Int | Var String | Op o [Exp o]
    deriving (Show)
data Ops = Add | Mul | Div | Sub | Sim
    deriving (Show)

instance Read (Exp Ops) where
    readsPrec _ s = readsExp s

readsExp :: ReadS (Exp Ops)
readsExp s = [((check op [a,b]),p5)
            |   ("(",p1) <- lex s,
                (a,p2) <- readsExp p1,
                ([op],p3) <- lex p2,
                op == '+' || op == '*' || op == '/' || op == '-',
                (b,p4) <- readsExp p3,
                (")",p5) <- lex p4]
            ++  [((Const ((read a)::Int)),sx) | (a,sx) <- lex s,all isDigit a]
            ++  [((Var a),sx) | (a,sx) <- lex s, all isAlpha a]
    where
    check op = case op of
              '+' -> Op Add
              '-' -> Op Sub
              '~' -> Op Sim
              '*' -> Op Mul
              '/' -> Op Div
