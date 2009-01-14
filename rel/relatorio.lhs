\documentclass[11pt,a4paper]{article}
\usepackage{graphicx}
\usepackage{alltt}
\usepackage[portuges]{babel}
\usepackage[latin1]{inputenc}
\usepackage{color}
\usepackage[portuges]{minitoc}
\usepackage{fancyhdr}
\usepackage{listings}
\usepackage{multicol}

\setlength{\textwidth}{16.5cm}
\setlength{\textheight}{24cm}
\setlength{\parindent}{1em}
\setlength{\parskip}{0pt plus 1pt}
\setlength{\oddsidemargin}{0cm}
\setlength{\evensidemargin}{0cm}
\setlength{\topmargin}{-1.1cm}
\setlength{\headsep}{20pt}
\setlength{\columnsep}{1.5pc}
\setlength\columnseprule{.4pt}
\setlength\premulticols{6\baselineskip}
\pagestyle{fancy}

\definecolor{gray_ulisses}{gray}{0.55}
\definecolor{castanho_ulisses}{rgb}{0.71,0.33,0.14}
\definecolor{preto_ulisses}{rgb}{0.41,0.20,0.04}
\definecolor{green_ulises}{rgb}{0.2,0.75,0}

\lstdefinelanguage{HaskellUlisses}
{
	basicstyle=\ttfamily\scriptsize,
	%backgroundcolor=\color{yellow},
	%frameshape={RYRYNYYYY}{yny}{yny}{RYRYNYYYY}, %contornos... muito nice...
	sensitive=true,
	morecomment=[l][\color{gray_ulisses}\scriptsize]{--},
	morecomment=[s][\color{gray_ulisses}\scriptsize]{\{-}{-\}},
	morestring=[b]",
	stringstyle=\color{red},
	showstringspaces=false,
	numbers=left,
	firstnumber=\thelstnumber,
	numberstyle=\tiny,
	numberblanklines=true,
	showspaces=false,
	showtabs=false,
	xleftmargin=15pt,
	xrightmargin=-20pt,
	emph=
	{[1]
		FilePath,IOError,abs,acos,acosh,all,and,any,appendFile,approxRational,asTypeOf,asin,
		asinh,atan,atan2,atanh,basicIORun,break,catch,ceiling,chr,compare,concat,concatMap,
		const,cos,cosh,curry,cycle,decodeFloat,denominator,digitToInt,div,divMod,drop,
		dropWhile,either,elem,encodeFloat,enumFrom,enumFromThen,enumFromThenTo,enumFromTo,
		error,even,exp,exponent,fail,filter,flip,floatDigits,floatRadix,floatRange,floor,
		fmap,foldl,foldl1,foldr,foldr1,fromDouble,fromEnum,fromInt,fromInteger,fromIntegral,
		fromRational,fst,gcd,getChar,getContents,getLine,head,id,inRange,index,init,intToDigit,
		interact,ioError,isAlpha,isAlphaNum,isAscii,isControl,isDenormalized,isDigit,isHexDigit,
		isIEEE,isInfinite,isLower,isNaN,isNegativeZero,isOctDigit,isPrint,isSpace,isUpper,iterate,
		last,lcm,length,lex,lexDigits,lexLitChar,lines,log,logBase,lookup,map,mapM,mapM_,max,
		maxBound,maximum,maybe,min,minBound,minimum,mod,negate,not,notElem,null,numerator,odd,
		or,ord,otherwise,pi,pred,primExitWith,print,product,properFraction,putChar,putStr,putStrLn,quot,
		quotRem,range,rangeSize,read,readDec,readFile,readFloat,readHex,readIO,readInt,readList,readLitChar,
		readLn,readOct,readParen,readSigned,reads,readsPrec,realToFrac,recip,rem,repeat,replicate,return,
		reverse,round,scaleFloat,scanl,scanl1,scanr,scanr1,seq,sequence,sequence_,show,showChar,showInt,
		showList,showLitChar,showParen,showSigned,showString,shows,showsPrec,significand,signum,sin,
		sinh,snd,span,splitAt,sqrt,subtract,succ,sum,tail,take,takeWhile,tan,tanh,threadToIOResult,toEnum,
		toInt,toInteger,toLower,toRational,toUpper,truncate,uncurry,undefined,unlines,until,unwords,unzip,
		unzip3,userError,words,writeFile,zip,zip3,zipWith,zipWith3
	},
	emphstyle={[1]\color{blue}},
	emph=
	{[2]
		Bool,Char,Double,Either,Float,IO,Integer,Int,Maybe,Ordering,Rational,Ratio,ReadS,ShowS,String
	},
	emphstyle={[2]\color{castanho_ulisses}},
	emph=
	{[3]
		case,class,data,deriving,do,else,if,import,in,infixl,infixr,instance,let,
		module,of,primitive,then,type,where
	},
	emphstyle={[3]\color{preto_ulisses}\textbf},
	emph=
	{[4]
		quot,rem,div,mod,elem,notElem,seq
	},
	emphstyle={[4]\color{castanho_ulisses}\textbf},
	emph=
	{[5]
		EQ,False,GT,Just,LT,Left,Nothing,Right,True,Show,Eq,Ord,Num
	},
	emphstyle={[5]\color{preto_ulisses}\textbf}
}

\lstnewenvironment{code}
{\textbf{Código Haskell} \hspace{1cm} \hrulefill \lstset{language=HaskellUlisses}}
{\hrule\smallskip}

%% stuff do minitoc %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\setcounter{secttocdepth}{10}
\setlength{\stcindent}{24pt}
\renewcommand{\stcfont}{\small\rm}
\renewcommand{\stcSSfont}{\small\bf}
%\newenvironment{mtc}{\secttoc\sectlof\sectlot}{\pagebreak}
%                        ^       ^        ^
%                    conteudos  figuras  tabelas
\newenvironment{mtc}{\secttoc\sectlof}{\pagebreak}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\title{\sf  Métodos de Programação I \\
\begin{tabular}{c}
	\includegraphics[width=.1\textwidth]{stuff/uminho.jpg}
	\includegraphics[width=.07\textwidth]{stuff/informatica.jpg}\\
	{\small Universidade do Minho}, {\small LESI}\\
	{\small Ano lectivo 2006/2007}\\
	{\small Trabalho Prático N$º$1}\\
\end{tabular}
}
\author{
	{\small Mário Ulisses Pires Araujo Costa - 43175} \and
	{\small Rami Galil Shashati - 43166} \and
	{\small Vasco Almeida Ferreira - 43207}}
\date{{\small \today}}

\begin{document}

\maketitle

\begin{abstract}
Este trabalho foi implementado no paradigma funcional, em linguagem \textbf{Haskell}.\\
O objectivo é desenvolver um compilador para uma pequena linguagem de programação.
\end{abstract}

\dosecttoc
%\dosectlof
\doparttoc
%\dopartlof
\tableofcontents
%\listoffigures
\pagebreak

\section{Introdu\c c\~ao}
\begin{code}
{-# OPTIONS -fglasgow-exts #-}
--
--  Métodos de Programação I
--  LESI, Universidade do Minho
--
--  Compilador para a linguagem Prog
--
--  2006/2007
--
--  autores:
--    Mário Ulisses Pires Araujo Costa - 43185
--    Rami Galil Shashati              - 43166
--    Vasco Almeida Ferreira           - 43207
--

module Trabalho1 where

import Data.Map
import Control.Monad.State
import Control.Monad.Error
import Data.List
import Maybe
import Char
import IO

infixr 5 >|<

\end{code}

Este Trabalho foi realizado no âmbito da disciplina de Métodos de Programação I, da Licenciatura em Engenharia de Sistemas e
Informática da Universidade do Minho.\\
O Objectivo deste trabalho é desenvolver um compilador para uma pequena linguagem de programação designada \textbf{Prog}.\\
Este relatório contem as diferentes fases de desenvolvimento do aplicação, a saber: análise do problema, escolha/descoberta
de algoritmos, implementação, testes e documentação.\\

\section{An\'alise e Especifica\c c\~ao}
\begin{mtc}
\subsection{Descri\c c\~ao informal do problema}
Nesta linguagem só existem três instruções sobre expressões. Estas podem ser constituídas por variáveis, inteiros
ou ambos.
Sempre que neste relatório nos referirmos a \textit{expressões} deve ser considerada a definição acima.\\
Lista das instruções aceites pelo programa e sua breve descrição:
\begin{itemize}
 \item \textbf{let $x$ = $e$} , onde $x$ é uma variável e $e$ uma expressão.
 \item \textbf{print $e$}, que imprime a expressão $e$.
 \item \textbf{if $exp$ then $p_1$ else $p_2$}, onde $exp$ é uma expressão lógica sobre variáveis e/ou inteiros e $p_1$
                                                    e $p_2$ ou são \textbf{print's} ou são \textbf{let's}.
\end{itemize}

\subsection{Especifica\c c\~ao dos Requisitos}
Para levar a cabo este trabalho, foram necessários conhecimentos da linguagem \textbf{Haskell}, do funcionamento
da linguagem \textbf{MSP} assim como o domínio de \LaTeXe{} e alguns dos seus pacotes para a elaboração deste relatório.

\subsubsection{Requisitos fundamentais}
Na base de desenvolvimento deste trabalho estão presentes duas ideias fundamentais. A primeira delas, consiste na "passagem" da linguagem 
descrita informalmente no item acima para outra linguagem chamada \textbf{MSP}. A esta passagem iremos, a partir de agora, informalmente, 
apelidar de \textit{compilação}. O segundo ponto consiste na "transformação" do resultado da \textit{compilação} numa \textit{computação}. Esta 
fase, por sua vez, será denominada por \textit{execução de um programa \textbf{MSP}}.
\end{mtc}

\section{Concep\c c\~ao/Desenho da Resolu\c c\~ao}
\begin{mtc}
\subsection{ExecProg}

Todo o código listado abaixo irá ser necessário para que consigamos usar a função $(>|<)$, que é a base para a resolução do nosso problema.\\

\begin{code}
main :: IO()
main = do
    putStr "Escreva o nome de um ficheiro: "
    f <- getLine
    ff <- readFile f
    let prog = (read ff :: Prog Ops)
    (>|<) compile execMSP prog
    putStr "prima qualquer tecla para mostrar o codigo msp: "
    c <- getChar
    case compile prog of
        (Right a) -> putStrLn $ show a
        (Left a)  -> putStrLn a
    return()

(>|<) :: (Eq o ,Opt o)
         => (Prog o -> Either String (MSP o))
         -> (MSP o -> IO ())
         -> Prog o -> IO ()
(>|<) comp exec prog = case comp prog of
                          (Left e)  -> print e
                          (Right m) -> exec m
\end{code}

\subsection{Estruturas de Dados e respectivas instâncias}
De seguida, vamos mostrar as estruturas de dados base deste trabalho e suas instâncias, nomeadamente;

\subsubsection{Prog o}
\begin{code}
data Prog o = If (Exp o) (Prog o) (Prog o)
            | Print (Exp o)
            | Let String (Exp o)
            | Seq (Prog o) (Prog o)
            | Ciclo (Prog o) (Exp o) (Prog o) (Prog o)
            | Prints String

progExemplo :: Prog Ops
progExemplo = Seq
                 (Seq
                    (Seq
                        (Seq
                            (Seq
                                (Seq
                                    (Let "wee" (Const 10))
                                    (Let "weee" (Const 20))
                                )
                                (If (Op AND_ [Var "x",Var "y"])
                                    (Print (Var "x"))
                                    (Print (Const 90)))
                            )
                            (Let "z" (Op Div [Const 2,Op Mul [Op Add [Var "x",Var "y"],Const 5]]))
                        )
                        (Print (Var "z"))
                    )
                    (If (Op AND_ [Var "x",Var "y"])
                        (Print (Const 101010))
                        (Print (Var "y"))))
                (Ciclo (Let "i" (Const 0)) (Op LE_ [Var "i",Const 10]) (Print (Var "i")) (Let "i" (Op Add [(Var "i"), (Const 1)])))
\end{code}

Acima mostramos a representação de um exemplo pertencente a (Porg o).\\

O exemplo seguinte corresponde ao modo como a linguagem deve ser escrita por um programador \textbf{Prog} num ficheiro texto.\\
\begin{footnotesize}
\ttfamily
let wee = 10;\\
let weee = 20;\\
if (x \&\& y) then print x else print 90;\\
let z = (2/((x+y)*5));\\
print z;\\
if (x \&\& y) then print 101010 else print y;\\
\[let i = 0 ; (i <= 10) ; let i = (i + 1)\]
    {print i;}
\end{footnotesize}

A unica classe que necessitamos para a realizaçao do trabalho é a classe Opt. Esta obriga tudo o que seja instância dela
a responder as funções nela definidas.

\begin{code}
class Opt o where
  arity :: o -> Int
  func  :: o -> ([Int] -> Int)

instance Opt Ops where
    arity Add = 2
    arity Mul = 2
    arity Sim = 1
    arity Sub = 2
    arity Div = 2

    arity OR_  = 2
    arity AND_ = 2
    arity NOT_ = 1
    arity GE_  = 2
    arity GT_  = 2
    arity LE_  = 2
    arity LT_  = 2
    arity EQ_  = 2
    arity NE_  = 2

    func Add = \[x,y] -> x + y
    func Mul = \[x,y] -> x * y
    func Sim = \[x]   -> -x
    func Sub = \[x,y] -> x - y
    func Div = \[x,y] -> x `div` y

    func OR_  = \[x,y] -> if (x /= 0) || (y /= 0) then 1 else 0
    func AND_ = \[x,y] -> if (x /= 0) && (y /= 0) then 1 else 0
    func NOT_ = \[x]   -> if x == 0 then 1 else 0
    func GE_  = \[x,y] -> if x >= y then 1 else 0
    func GT_  = \[x,y] -> if x >  y then 1 else 0
    func LE_  = \[x,y] -> if x <= y then 1 else 0
    func LT_  = \[x,y] -> if x <  y then 1 else 0
    func EQ_  = \[x,y] -> if x == y then 1 else 0
    func NE_  = \[x,y] -> if x /= y then 1 else 0
\end{code}

Instância de Show para o tipo de dados Prog o.\\

\begin{code}
instance (Opt o, Show o) => Show (Prog o) where
    show (Print a)    = "print " ++ show a
    show (Let s a)    = "let " ++ s ++ " = " ++ show a
    show (Seq l r)    = show l ++ show r
    show (If exp p1 p2) = "if (" ++ show exp ++ ")" ++
                           "then (" ++ show p1 ++ ")" ++
                           "else (" ++ show p2 ++ ");\n"
    show (Ciclo inic exp c1 fim) = "[ " ++ show inic ++
                         ", " ++ show exp ++ ";" ++
                         show fim ++ "]\n\t" ++ show c1 ++ "\n"
    show (Prints e)     = "prints " ++ show e ++ "\n"
\end{code}

É apresentada agora a solução para a instância \textbf{Read} do data \textbf{Prog Ops};\\

\begin{code}
instance Read (Prog Ops) where
    readList = reads' . readsProg
    readsPrec _ s =
        let
            removeComents :: String -> String
            removeComents = let remove l = [x | x <- l , (fst $ (\[l] -> l) x) /= "#"]
                            in init . unlines . Prelude.map unwords . concat .
                               Prelude.map (\(a,b) -> [a++b]) . Prelude.map unzip .
                               remove . Prelude.map lex . lines
            read' :: String -> Prog Ops
            read' = ins . reverse . fst . (\[l] -> l) . reads' . readsProg
        in [(read' $ removeComents s,[])]


ins :: [Prog Ops] -> Prog Ops
ins [e] = e
ins (h:t) = Seq (ins t) h

reads' :: [([Prog Ops],String)] -> [([Prog Ops],String)]
reads' [(a,b)] = [(a++c,fim) | (c,fim) <- readsProg b]

readsProg :: ReadS [Prog Ops]
readsProg s | s == []   = [([],[])]
            | otherwise = [(h:t,rest) | ("let",p1)   <- lex s,
                                        (h,fim)      <- readsProgLetPrint s,
                                        (t,rest)     <- readsProg fim        ] ++
                          [(h:t,rest) | ("print",p1) <- lex s,
                                        (h,p2)       <- readsProgLetPrint s,
                                        (t,rest)     <- readsProg p2         ] ++
            [((If exp_ h t):tt ,rest) | ("if",p1)    <- lex s,
                                        (exp_,p2)    <- readsExp p1,
                                        ("then",p3)  <- lex p2,
                                        (h,p4)       <- readsProgThen p3,
                                        ("else",p5)  <- lex p4,
                                        (t,p6)       <- readsProgLetPrint p5,
                                        (tt,rest)    <- readsProg p6         ] ++
 [([(Ciclo i cond (ins p9) inc)],p10) |
                                        ("[",p1)     <- lex s,
                                        (i,p2)       <- readsProgThen p1,
                                        (";",p3)     <- lex p2,
                                        (cond,p4)    <- readsExp p3,
                                        (";",p5)     <- lex p4,
                                        (inc,p6)     <- readsProgThen p5,
                                        ("]",p7)     <- lex p6,
                                        ("{",p8)     <- lex p7,
                                        (p9,p10)     <- readsProg p8         ] ++
                             [([],p1) | ("}",p1)     <- lex s                ]
    where
    readsProgLetPrint :: ReadS (Prog Ops)
    readsProgLetPrint s =
        [(Print exp_,rest)    | ("print",p1) <- lex s,
                                (exp_,p2)    <- readsExp p1,
                                (";",rest)   <- lex p2        ] ++
        [(Let val exp2_,rest) | ("let",p1)   <- lex s,
                                (val,p2)     <- lex p1,
                                ("=",exp_)   <- lex p2,
                                (exp2_,p3)   <- readsExp exp_,
                                (";",rest)   <- lex p3        ]
    readsProgThen :: ReadS (Prog Ops)
    readsProgThen s =
        [(Print exp_,rest)    | ("print",p1) <- lex s,
                                (exp_,rest)  <- readsExp p1   ] ++
        [(Let val exp2_,rest) | ("let",p1)   <- lex s,
                                (val,p2)     <- lex p1,
                                ("=",exp_)   <- lex p2,
                                (exp2_,rest) <- readsExp exp_ ]
\end{code}

\subsubsection{Exp o}
\begin{code}
data Exp o = Const Int
           | Var String
           | Op o [Exp o]

instance (Show o,Opt o) => Show (Exp o) where
    show (Const n) = show n
    show (Var s)   = s
    show (Op o l)  | arity o == 2 = "(" ++ (show $ head l) ++ show o ++
                                    (show $ last l) ++ ")"
                   | arity o == 1 = "(" ++ show o ++ (show $ head l) ++ ")"

instance Read (Exp Ops) where
    readsPrec _ s = readsExp s

readsExp :: ReadS (Exp Ops)
readsExp s =
    [((leOp "~" [a]),p4) |  ("(",p1)  <- lex s,
                            ("~",p2)  <- lex p1,
                            (a,p3)    <- readsExp p2,
                            (")",p4)  <- lex p3] ++
    [((leOp op [a,b]),p5) | ("(",p1)  <- lex s,
                            (a,p2)    <- readsExp p1,
                            (op,p3)   <- lex p2,
                            op == "+"  || op == "*"  || op == "/"  ||
                            op == "-"  || op == "||" || op == "&&" ||
                            op == "==" || op == "!=" || op == ">=" ||
                            op == "<=" || op == "!"  || op == ">"  ||
                            op == "<",
                            (b,p4)   <- readsExp p3,
                            (")",p5) <- lex p4                    ] ++
    [((Const ((read a)::Int)),sx) | (a,sx) <- lex s, all isDigit a] ++
    [((Var a),sx)                 | (a,sx) <- lex s, all isAlpha a]
    where
    leOp :: String -> [Exp Ops] -> Exp Ops
    leOp o = Op (read o::Ops)

\end{code}

\subsubsection{Ops}
\begin{multicols}{2}
\begin{code}
data Ops = Add
         | Mul
         | Sim
         | Sub
         | Div
         | OR_
         | AND_
         | NOT_
         | GE_
         | GT_
         | LE_
         | LT_
         | EQ_
         | NE_
    deriving Eq

instance Show Ops where
    show Add = "+"
    show Mul = "*"
    show Sim = "~"
    show Sub = "-"
    show Div = "/"

    show OR_  = "||"
    show AND_ = "&&"
    show NOT_ = "!"
    show GE_  = ">="
    show GT_  = ">"
    show LE_  = "<="
    show LT_  = "<"
    show EQ_  = "=="
    show NE_  = "!="

instance Read Ops where
    readsPrec _ "+"  = [(Add,[]) ]
    readsPrec _ "-"  = [(Sub,[]) ]
    readsPrec _ "*"  = [(Mul,[]) ]
    readsPrec _ "/"  = [(Div,[]) ]
    readsPrec _ "~"  = [(Sim,[]) ]
    readsPrec _ "||" = [(OR_,[]) ]
    readsPrec _ "&&" = [(AND_,[])]
    readsPrec _ "!"  = [(NOT_,[])]
    readsPrec _ ">=" = [(GE_,[]) ]
    readsPrec _ ">"  = [(GT_,[]) ]
    readsPrec _ "<=" = [(LE_,[]) ]
    readsPrec _ "<"  = [(LT_,[]) ]
    readsPrec _ "==" = [(EQ_,[]) ]
    readsPrec _ "!=" = [(NE_,[]) ]
\end{code}
\end{multicols}

\subsubsection{Instr o}
\begin{code}
data Instr o = PUSH Int
             | LOAD
             | STORE
             | IN
             | OUT
             | OP o
             | JMP String
             | JMPF String
             | HALT
             | Label String
             | OUTC
    deriving Eq

instance (Show o) => Show (Instr o) where
    show (PUSH n)  = "\tPUSH " ++ show n  ++ "\n"
    show (OP op)   = "\tOP "   ++ show op ++ "\n"
    show (LOAD)    = "\tLOAD"  ++ "\n"
    show (STORE)   = "\tSTORE" ++ "\n"
    show (IN)      = "\tIN"    ++ "\n"
    show (OUT)     = "\tOUT"   ++ "\n"
    show (HALT)    = "\tHALT"  ++ "\n"
    show (JMP s)   = "\tJMP "  ++ s ++ "\n"
    show (JMPF s)  = "\tJMPF " ++ s ++ "\n"
    show (Label s) = "\t "     ++ s ++ "\n"

    showList [] _ = []
    showList (h:t) _ = show h ++ showList t []
\end{code}

\subsubsection{Tipos de dados auxiliares}
\begin{code}
data Mem = Mem {stack :: [Int] , heap :: [Maybe Int]}
type MSP o = [Instr o]
type VarDict = Map String Int
\end{code}

\subsection{Estrutura da Aplica\c c\~ao}
O compilador de \textbf{Prog} tem como base duas funções fundamentais, a saber:\\

\subsubsection{Compilação}
A função \textbf{compile} tem como objectivo a "passagem" do tipo de dados \textbf{Prog o} para \textbf{MSP o}.

Esta função tem contradomínio \textit{Either String (MSP o)} pelo simples facto dela gerar código \textbf{MSP o} correcto
e pronto a ser executado pela função \textbf{execMSP} propagar um erro de compilação (atraves do uso de uma String).\\

A função auxiliar à \textit{compile}, \textit{compile\_} tem contradomínio \textit{State VarDict (Either String (MSP o))} para garantir
que mantemos o nosso dicionário de variáveis associadas a valores sempre actualizado. Estas variáveis nada mais são do que as String's
que são usadas aquando da declaração das mesmas na linguagem \textbf{Prog}, por exemplo:\\
\begin{footnotesize}
\ttfamily
let n1 = 10;\\
let n2 = 20;\\
let n3 = 30;\\
\end{footnotesize}

Note-se que o tipo de dados de VarDict corresponde ao mapeamento de String com o respectivo valor.\\

A melhor estratégia que encontramos para propagar erros foi através do operador de combinador de monads (>>=).
Com isto deixamos de ter a preocupação das verificações exaustivas de computações falhadas. Para isto usamos uma função fundamental
que avalia as expressões, retornando Left \textit{Erro} em caso de insucesso e Right \textit{MSP o} caso contrário.

De seguida lista-se o código referente à função acima descrita:\\
\begin{code}

compile  :: (Eq o ,Opt o) => Prog o -> Either String (MSP o)
compile p = evalState (compile' p) empty
    where
    compile' :: (Eq o , Opt o) => Prog o -> State VarDict (Either String (MSP o)) 
    compile' (Print e) = do vd <- get
                            return (evalExp vd e >>= (\x -> return (x ++ [OUT])))
    compile' (Let x e) = do vd <- get
                            if (Data.Map.member x vd)
                                then return ()
                                else put $ Data.Map.insert x (succ (Data.Map.size vd)) vd
                            vd' <- get
                            return (evalExp vd e
                                    >>= (\m -> return ([PUSH (vd' ! x)] ++
                                                       m ++ [STORE])))
    compile' (Seq x y) = compile' x >>= (\x -> compile' y >>=
                         (\y -> return (x >>= (\x1 -> y >>=
                         (\y1 -> return (x1 ++ y1))))))
    compile' (If exp_ p1 p2) = do vd <- get
                                  let evale = evalExp vd exp_
                                  compile' p1 >>= (\x -> compile' p2 >>=
                                    (\y ->  return (x  >>= (\then_ -> y >>=
                                    (\else_ -> evale >>=
                                    (\eval -> return (eval  ++ [JMPF ("#THEN")] ++
                                                      then_ ++ [(JMP "#FIM")]   ++
                                                      else_ ++ [(Label "#FIM")])))))))
    compile' (Ciclo (Let x e) exp_ c (Let _ e1)) =
        do vd <- get
           put (Data.Map.insert x 1024 vd)
           vd <- get
           compile' c >>= (\cicl -> return (cicl >>= (\ciclo -> evalExp vd exp_ >>=
             (\cond -> evalExp vd e >>= (\exp1 -> evalExp vd e1 >>=
               (\exp2 -> return ( [PUSH 1024] ++ exp1 ++ [STORE] ++ [Label "#CICLO"] ++
                                  cond ++ [JMPF "#FIM"] ++ ciclo ++ [PUSH 1024]++ exp2 ++
                                  [STORE] ++ [JMP "#CICLO"])))))))
    compile' (Prints s) = return $ return $ concat $ Prelude.map (\c -> [PUSH (ord c)] ++ [OUTC]) s

evalExp            :: Opt o => VarDict -> Exp o -> Either String (MSP o)
evalExp _ (Const x) = return [PUSH x]
evalExp vd (Var x)  | (Data.Map.member x vd) = return ([PUSH (vd ! x)] ++ [LOAD])
                    | otherwise = Left ("!! Erro Compilacao -> Variavel " ++ x ++ " nao declarada!")
evalExp vd (Op o l) | arity o == 1 = evalExp vd (head l) >>= (\x -> return (x ++ [OP o]))
                    | arity o == 2 = let eval1 = evalExp vd (head l) 
                                         eval2 = evalExp vd (l!!1)
                                     in eval1 >>= (\x -> eval2 >>= (\y -> return (x ++ y ++ [OP o])))
\end{code}

\subsubsection{Execução}

\begin{footnotesize}
\ttfamily
aux :: (Eq o,Show o,Opt o) => MSP o -> StateT Mem (ErrorT String IO) ()
\end{footnotesize}\\

O objectivo da \textbf{aux} é conseguirmos um transformador de monads tanto de State como de IO como de Erro.\\
Através do uso do monad StateT conseguimos ir actualizando a memória do compilador.\\
Já o monad responsável pela propagação de erros, neste caso monad ErrorT, possibilita-nos a utilização da função \textit{throwError}
que nos permite pegar numa String (um erro) e encapsulá-lo dentro de um monad.\\
O uso do monad IO faz com que possamos imprimir os resultados que vamos obtendo, quer estes sejam erros ou resultados válidos provenientes
de uma computação.\\

A nossa função que usa a \textit{aux} e que executa código (MSP o) tem como assinatura:\\

\begin{footnotesize}
\ttfamily
execMSP :: (Eq o , Show o ,Opt o) => MSP o -> IO ()
\end{footnotesize}\\

Ou seja, pega num (MSP o) e executa a função runErrorT que devolve um (m Either e a) que irá ser fundamental para a função acima descrita.\\

De seguida lista-se o código referente à função acima descrita:\\
\begin{code}
execMSP    :: (Eq o , Show o ,Opt o) => MSP o -> IO ()
execMSP msp = do eith <- runErrorT (evalStateT (aux msp) emptymem)
                 case eith of
                     (Left a)  -> print a
                     (Right b) -> if b /= ()
                                  then print b
                                  else putStr ""
    where
    aux      :: (Eq o,Show o,Opt o) => MSP o -> StateT Mem (ErrorT String IO) ()
    aux []    = return ()
    aux (h:t) =
      case h of
        (PUSH x) -> do mem <- get
                       put $ pushOrin mem x
                       aux t
        (IN)     -> do mem <- get
                       x   <- lift $ lift $ getLine
                       put $ pushOrin mem $ (read x :: Int)
                       aux t
        (STORE)  -> do mem <- get
                       case store mem of
                         (Left e)  -> throwError e
                         (Right m) -> do put m
                                         aux t
        (LOAD)   -> do mem <- get
                       case load mem of
                         (Left e)  -> throwError e
                         (Right m) -> do put m
                                         aux t
        (OUT)    -> do mem <- get
                       case checkMem mem 1 of
                         False ->
                           do let err = "!! Erro Execucao -> Out: not enough arguments for function out!"
                              throwError err
                         True  -> do lift $ lift $ putStrLn $ show $ head $ stack mem
                                     put $ out mem
                                     aux t
        (OP o)   -> do mem <- get
                       case op mem o of
                         (Left e)  -> throwError e
                         (Right m) -> do put m
                                         aux t
        (JMPF s) -> do mem <- get
                       case checkMem mem 1 of
                         False ->
                           do let err = "!! Erro Execucao -> Jmpf: not enough arguments for function JMPF!"
                              throwError err
                         True  -> if ((head $ stack mem) /= 0)
                                    then aux t
                                    else aux $ tail $ dropWhile (/=(JMP "#FIM")) t
                       put(out mem)
        (HALT)   -> return()
        (JMP "#FIM")  -> do aux $ tail $ dropWhile (/=(Label "#FIM")) t
        (Label "#CICLO") -> do aux $ takeWhile (/=(JMPF "#FIM")) t
                               mem <- get
                               if (head $ stack mem) == 0
                                 then aux $ tail $ dropWhile (/=(JMP "#CICLO")) t
                                 else do aux $ takeWhile (/= (JMP "#CICLO")) $
                                           tail $ dropWhile (/=(JMPF "#FIM")) t
                                         aux (h:t)
        (Label s) -> do aux t
        (OUTC)    -> do mem <- get
                        case checkMem mem 1 of
                          False ->
                            do let err = "!! Erro Execucao -> OutC: not enough arguments for function out!"
                               throwError err
                          True  -> do lift $ lift $ putStr $ (:) (chr $ head $ stack mem) []
                                      put $ out mem
                                      aux t
\end{code}

\subsubsection{Memória}
\begin{code}
emptymem :: Mem
emptymem  = Mem {stack = [], heap = replicate 1024 Nothing}
\end{code}

A memória sendo representada por uma lista de inteiros á qual designamos stack e uma lista de Maybe Int chamada heap, necessita
das seguintes funções básicas que vão trabalhar sobre ela, nestas temos já em atenção muitos dos possíveis erros que podem surgir
quando se executa \textbf{MSP}.\\

\begin{code}
injh :: Mem -> [Maybe Int] -> Mem
injh  = (\mem l -> Mem {stack = stack mem, heap = l})

injs :: Mem -> [Int] -> Mem
injs  = (\mem l -> Mem {stack = l, heap = heap mem})

pushOrin      :: Mem -> Int -> Mem
pushOrin mem i = injs mem $ (:) i $ stack mem

store    :: Mem -> Either String Mem
store mem | not(checkMem mem 2) = Left "!! Erro Execucao -> Store: not enough arguments for function store!"
          | otherwise           = Right (injh mem' ((take (e - 1) hm) ++ [Just v] ++ (drop e hm)))
    where
      hm   = heap mem
      sm   = stack mem
      v    = head sm
      e    = head $ tail sm
      mem' = injs mem $ drop 2 sm

load    :: Mem -> Either String Mem
load mem | not(checkMem mem 1) = Left "!! Erro Execucao -> Load: not enough arguments for function load!"
         | (heap mem !! (end-1)) == Nothing = Left "!! Erro Execucao -> Load: variavel nao declarada!"
         | otherwise = Right (injs mem $ fromJust (b !! (end - 1)) : t)
    where
       s   = stack mem
       end = head s
       t   = tail s
       b   = heap mem

out    :: Mem -> Mem
out mem = (injs mem $ tail $ stack mem)

op       :: (Opt o, Show o, Eq o) => Mem -> o -> Either String Mem
op  mem o | not(checkMem mem ar) = Left "!! Erro Execucao -> Op: not enough arguments for function op!"
          | otherwise = if (show o == "/") && ((last $ reverse $ tk) == 0)
                          then Left "!! Erro Execucao -> Op: Div 0 !"
                          else Right (injs mem ((func o $ reverse $ tk) : dr))
     where
       ar = arity o
       s  = stack mem
       dr = drop ar s
       tk = take ar s

\end{code}
\end{mtc}

\section{Testes}
\begin{mtc}
\subsection{Exemplos de prog's e msp's}
\begin{code}

mspprog' = [PUSH 6, PUSH 3, OP Div, PUSH 1024, PUSH 0, STORE, Label "#CICLO", PUSH 1024, LOAD, PUSH 15000,OP LE_, JMPF "#FIM", PUSH 1024,LOAD, OUT, PUSH 1024, PUSH 1024, LOAD, PUSH 1 , OP Add, STORE, JMP "#CICLO", PUSH 30303030, OUT]

checkMem :: Mem -> Int -> Bool
checkMem mem n = let stck = stack mem
                 in length stck >= n

exp1 :: Exp Ops
exp1  = Op Add [Const 3 , Op Mul [Const 2 ,Var "x"]]

msp :: MSP Ops
msp  = [PUSH 3, PUSH 3, STORE, PUSH 3, LOAD, OUT,PUSH 99, OUTC,PUSH 99, OUTC,
        PUSH 99, OUTC,PUSH 99, OUTC,PUSH 99, OUTC,PUSH 99, OUTC,PUSH 99, OUTC,
        PUSH 99, OUTC,PUSH 99, OUTC,PUSH 99, OUTC,PUSH 99, OUTC ]

prog''  = Seq (Seq (Seq (Seq (Seq (Let "x" (Const 10))
                                (Let "y" (Const 20)))
                           (If (Op AND_ [Var "x",Const 34])
                               (Print (Var "x"))
                               (Print (Const 90))))
                      (Let "y" (Op Add [Var "x", Var "y"])))
                 (Ciclo (Let "i" (Const 0))
                        (Op LE_ [Var "i",Const 10])
                        (Print (Var "i"))
                        (Let "i" (Op Add [(Var "i"), (Const 1)]))))
            (If (Op AND_ [Var "x",Const 345])
                (Print (Const 101010))
                (Print (Var "y")))

progif = (If (Op EQ_ [(Const 1),(Const 1)]) (Print (Const 1)) (Print (Const 3)))

tt :: Prog Ops
tt = If (Op OR_ [Var "123",Var "345"]) (Print (Const 4)) (Print (Const 3))

prog :: Prog Ops
prog  = Seq (Seq (Seq (Seq (Seq (Let "x" (Const 10))
                           (Let "y" (Const 20)))
                      (Let "z" (Op Add [Var "x", Var "y"])))
                 (If (Op AND_ [Var "z",Const 345]) (Print (Const 1)) (Print (Const 3))))
            (Print (Var "z")))
            (Prints "mp1 a melhor cadeira de sempre! \n")

mspprog :: MSP Ops
mspprog  = [PUSH 2, PUSH 2, STORE, PUSH 3, OP Sim,
            PUSH 4, PUSH 2, LOAD, OP Sub, OP Mul, OUT]
\end{code}
\end{mtc}
\end{document}
