use crate::tokenize::*;
use crate::parse::*;

use std::fmt;
use std::error::Error as StdError;
use std::str::FromStr;

/// 字句解析エラーと構文解析エラーを統合するエラー型
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Error {
    Lexer(LexError),
    Parser(ParseError),
}

impl From<LexError> for Error {
    fn from(e: LexError) -> Self {
        Error::Lexer(e)
    }
}

impl From<ParseError> for Error {
    fn from(e: ParseError) -> Self {
        Error::Parser(e)
    }
}

impl FromStr for Ast {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // 内部では字句解析、構文解析の順に実行する
        let tokens = lex(s)?;
        let ast = parse(tokens)?;
        Ok(ast)
    }
}

impl fmt::Display for Loc {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}-{}", self.0, self.1)
    }
}


impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::TokenKind::*;
        match self {
            Number(n) => n.fmt(f),
            Plus => write!(f, "+"),
            Minus => write!(f, "-"),
            Asterisk => write!(f, "*"),
            Slash => write!(f, "/"),
            LParen => write!(f, "("),
            RParen => write!(f, ")"),
            Lt => write!(f, "<"),
            Gt => write!(f, ">"),
            Eq => write!(f, "=="),
            Neq => write!(f, "!="),
            Le => write!(f, "<="),
            Ge => write!(f, ">="),
        }
    }
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::LexErrorKind::*;
        let loc = &self.loc;
        match self.value {
            InvalidChar(c) => write!(f, "{}: invalid char '{}'", loc, c),
            Eof => write!(f, "End of file"),
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::ParseError::*;
        match self {
            UnexpectedToken(tok) => write!(f, "{}: {} is not expected", tok.loc, tok.value),
            NotExpression(tok) => write!(
                f,
                "{}: '{}' is not a start of expression",
                tok.loc, tok.value
            ),
            NotOperator(tok) => write!(f, "{}: '{}' is not an operator", tok.loc, tok.value),
            UnclosedOpenParen(tok) => write!(f, "{}: '{}' is not closed", tok.loc, tok.value),
            RedundantExpression(tok) => write!(
                f,
                "{}: expression after '{}' is redundant",
                tok.loc, tok.value
            ),
            Eof => write!(f, "End of file"),
        }
    }
}

impl StdError for LexError {}

impl StdError for ParseError {}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        use self::Error::*;
        match self {
            Lexer(lex) => Some(lex),
            Parser(parse) => Some(parse),
        }
    }
}

/// `input` に対して `loc` の位置を強調表示する
fn print_annot(input: &str, loc: Loc) {
    // 入力に対して
    eprintln!("{}", input);
    // 位置情報をわかりやすく示す
    eprintln!("{}{}", " ".repeat(loc.0), "^".repeat(loc.1 - loc.0));
}

impl Error {
    /// 診断メッセージを表示する
    pub(crate) fn show_diagnostic(&self, input: &str) {
        use self::Error::*;
        use self::ParseError as P;
        // エラー情報とその位置情報を取り出す。エラーの種類によって位置情報を調整する。
        let (e, loc): (&dyn StdError, Loc) = match self {
            Lexer(e) => (e, e.loc.clone()),
            Parser(e) => {
                let loc = match e {
                    P::UnexpectedToken(Token { loc, .. })
                    | P::NotExpression(Token { loc, .. })
                    | P::NotOperator(Token { loc, .. })
                    | P::UnclosedOpenParen(Token { loc, .. }) => loc.clone(),
                    // redundant expressionはトークン以降行末までが余りなのでlocの終了位置を調整する
                    P::RedundantExpression(Token { loc, .. }) => Loc(loc.0, input.len()),
                    // EoFはloc情報を持っていないのでその場で作る
                    P::Eof => Loc(input.len(), input.len() + 1),
                };
                (e, loc)
            }
        };
        // エラー情報を簡単に表示し
        eprintln!("{}", e);
        // エラー位置を指示する
        print_annot(input, loc);
    }
}

pub fn show_trace<E: StdError>(e: E) {
    // エラーがあった場合そのエラーとcauseを全部出力する
    eprintln!("{}", e);
    let mut source = e.source();
    // cause を全て辿って表示する
    while let Some(e) = source {
        eprintln!("caused by {}", e);
        source = e.source()
    }
    // エラー表示のあとは次の入力を受け付ける
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "parser error")
    }
}
