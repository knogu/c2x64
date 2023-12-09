use std::error::Error as StdError;
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Loc(usize, usize);

impl Loc {
    fn merge(&self, other: &Loc) -> Loc {
        use std::cmp::{max, min};
        Loc(min(self.0, other.0), max(self.1, other.1))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Annot<T> {
    value: T,
    loc: Loc,
}

impl<T> Annot<T> {
    fn new(value: T, loc: Loc) -> Self {
        Self {value, loc}
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum TokenKind {
    Number(u64),
    Plus,
    Minus,
    Asterisk,
    Slash,
    LParen,
    RParen,
    Eq,
    Neq,
    Lt, // Less than
    Le, // Less or Equal
}

type Token = Annot<TokenKind>;

impl Token {
    fn number(n: u64, loc: Loc) -> Self {
        Self::new(TokenKind::Number(n), loc)
    }

    fn plus(loc: Loc) -> Self {
        Self::new(TokenKind::Plus, loc)
    }

    fn minus(loc: Loc) -> Self {
        Self::new(TokenKind::Minus, loc)
    }

    fn asterisk(loc: Loc) -> Self {
        Self::new(TokenKind::Asterisk, loc)
    }

    fn slash(loc: Loc) -> Self {
        Self::new(TokenKind::Slash, loc)
    }

    fn lparen(loc: Loc) -> Self {
        Self::new(TokenKind::LParen, loc)
    }

    fn rparen(loc: Loc) -> Self {
        Self::new(TokenKind::RParen, loc)
    }

    fn eq(loc: Loc) -> Self {
        Self::new(TokenKind::Eq, loc)
    }

    fn neq(loc: Loc) -> Self {
        Self::new(TokenKind::Neq, loc)
    }

    fn le(loc: Loc) -> Self {
        Self::new(TokenKind::Le, loc)
    }

    fn lt(loc: Loc) -> Self {
        Self::new(TokenKind::Lt, loc)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum LexErrorKind {
    InvalidChar(char),
    Eof,
}

type LexError = Annot<LexErrorKind>;

impl LexError {
    fn invalid_char(c: char, loc: Loc) -> Self {
        LexError::new(LexErrorKind::InvalidChar(c), loc)
    }

    fn eof(loc: Loc) -> Self {
        LexError::new(LexErrorKind::Eof, loc)
    }
}

/// `pos` のバイトが期待するものであれば1バイト消費して `pos`を1進める
fn consume_byte(input: &[u8], pos: usize, b: &[u8]) -> Result<(Vec<u8>, usize), LexError> {
    let size = b.len();
    // Check input length is eq or gt expected bytes size
    if input.len() + size - 1 <= pos {
        return Err(LexError::eof(Loc(pos, pos)));
    }
    // 入力が期待するものでなければエラー
    if input[pos..pos + size] != b[..size] {
        return Err(LexError::invalid_char(
            input[pos] as char,
            Loc(pos, pos + 1),
        ));
    }

    Ok((b[..size].to_vec(), pos + size))
}
fn recognize_many(input: &[u8], mut pos: usize, mut f: impl FnMut(u8) -> bool) -> usize {
    while pos < input.len() && f(input[pos]) {
        pos += 1;
    }
    pos
}

fn lex_number(input: &[u8], pos: usize) -> Result<(Token, usize), LexError> {
    use std::str::from_utf8;

    let start = pos;
    let end = recognize_many(input, start, |b| b"1234567890".contains(&b));
    let n = from_utf8(&input[start..end])
        .unwrap()
        .parse()
        .unwrap();
    Ok((Token::number(n, Loc(start, end)), end))
}

fn lex_plus(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    // `Result::map` を使うことで結果が正常だった場合の処理を簡潔に書ける。
    // これはこのコードと等価
    // ```
    // match consume_byte(input, start, b'+') {
    //     Ok((_, end)) => (Token::plus(Loc(start, end)), end),
    //     Err(err) => Err(err),
    // }
    consume_byte(input, start, b"+").map(|(_, end)| (Token::plus(Loc(start, end)), end))
}

fn lex_minus(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    consume_byte(input, start, b"-").map(|(_, end)| (Token::minus(Loc(start, end)), end))
}
fn lex_asterisk(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    consume_byte(input, start, b"*").map(|(_, end)| (Token::asterisk(Loc(start, end)), end))
}
fn lex_slash(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    consume_byte(input, start, b"/").map(|(_, end)| (Token::slash(Loc(start, end)), end))
}
fn lex_lparen(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    consume_byte(input, start, b"(").map(|(_, end)| (Token::lparen(Loc(start, end)), end))
}
fn lex_rparen(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    consume_byte(input, start, b")").map(|(_, end)| (Token::rparen(Loc(start, end)), end))
}

fn lex_lt(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    consume_byte(input, start, b"<").map(|(_, end)| (Token::lt(Loc(start, end)), end))
}

fn lex_eq(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    consume_byte(input, start, b"==").map(|(_, end)| (Token::eq(Loc(start, end)), end))
}

fn lex_neq(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    consume_byte(input, start, b"!=").map(|(_, end)| (Token::neq(Loc(start, end)), end))
}

fn lex_le(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    consume_byte(input, start, b"<=").map(|(_, end)| (Token::le(Loc(start, end)), end))
}

fn skip_spaces(input: &[u8], pos: usize) -> Result<((), usize), LexError> {
    let pos = recognize_many(input, pos, |b| b" \n\t".contains(&b));
    Ok(((), pos))
}

fn lex(input: &str) -> Result<Vec<Token>, LexError> {
    let mut tokens = Vec::new();
    let input = input.as_bytes();
    let mut pos = 0;

    // サブレキサを呼んだ後`pos`を更新するマクロ
    macro_rules! lex_a_token {
        ($lexer: expr) => {{
            let (tok, p) = $lexer?;
            tokens.push(tok);
            pos = p;
        }};
    }

    while pos < input.len() {
        if pos + 1 < input.len() {
            if &input[pos..pos+2] == b"==" {
                lex_a_token!(lex_eq(input, pos));
                continue
            }
            if &input[pos..pos+2] == b"!=" {
                lex_a_token!(lex_neq(input, pos));
                continue
            }
            if &input[pos..pos+2] == b"<=" {
                lex_a_token!(lex_le(input, pos));
                continue
            }
        }
        match input[pos] {
            // 遷移図通りの実装
            b'0'..=b'9' => lex_a_token!(lex_number(input, pos)),
            b'+' => lex_a_token!(lex_plus(input, pos)),
            b'-' => lex_a_token!(lex_minus(input, pos)),
            b'*' => lex_a_token!(lex_asterisk(input, pos)),
            b'/' => lex_a_token!(lex_slash(input, pos)),
            b'(' => lex_a_token!(lex_lparen(input, pos)),
            b')' => lex_a_token!(lex_rparen(input, pos)),
            b'<' => lex_a_token!(lex_lt(input, pos)),
            // 空白を扱う
            b' ' | b'\n' | b'\t' => {
                let ((), p) = skip_spaces(input, pos)?;
                pos = p;
            }
            // それ以外がくるとエラー
            b => return Err(LexError::invalid_char(b as char, Loc(pos, pos + 1))),
        }
    }

    Ok(tokens)
}

#[test]
fn test_lexer() {
    assert_eq!(
        lex("1 + 2 * 3 - -10"),
        Ok(vec![
            Token::number(1, Loc(0, 1)),
            Token::plus(Loc(2, 3)),
            Token::number(2, Loc(4, 5)),
            Token::asterisk(Loc(6, 7)),
            Token::number(3, Loc(8, 9)),
            Token::minus(Loc(10, 11)),
            Token::minus(Loc(12, 13)),
            Token::number(10, Loc(13, 15)),
        ])
    )
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum UniOpKind {
    Plus,
    Minus,
}

type UniOp = Annot<UniOpKind>;

impl UniOp {
    fn plus(loc: Loc) -> Self {
        Self::new(UniOpKind::Plus, loc)
    }

    fn minus(loc: Loc) -> Self {
        Self::new(UniOpKind::Minus, loc)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum BinOpKind {
    Add,
    Sub,
    Mult,
    Div,
    Eq,
    Neq,
    Lt,
    Le,
}

type BinOp = Annot<BinOpKind>;

impl BinOp {
    fn add(loc: Loc) -> Self {
        Self::new(BinOpKind::Add, loc)
    }

    fn sub(loc: Loc) -> Self {
        Self::new(BinOpKind::Sub, loc)
    }

    fn mult(loc: Loc) -> Self {
        Self::new(BinOpKind::Mult, loc)
    }

    fn div(loc: Loc) -> Self {
        Self::new(BinOpKind::Div, loc)
    }

    fn eq(loc: Loc) -> Self {
        Self::new(BinOpKind::Eq, loc)
    }

    fn neq(loc: Loc) -> Self {
        Self::new(BinOpKind::Neq, loc)
    }

    fn lt(loc: Loc) -> Self {
        Self::new(BinOpKind::Lt, loc)
    }

    fn le(loc: Loc) -> Self {
        Self::new(BinOpKind::Le, loc)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum AstKind {
    Num(u64),
    UniOp {op: UniOp, e: Box<Ast>},
    BinOp {op: BinOp, l: Box<Ast>, r: Box<Ast>},
}

type Ast = Annot<AstKind>;

impl Ast {
    fn num(n: u64, loc: Loc) -> Self {
        Self::new(AstKind::Num(n), loc)
    }

    fn uniop(op: UniOp, e: Ast, loc: Loc) -> Self {
        Self::new(AstKind::UniOp { op, e: Box::new(e) }, loc)
    }

    fn binop(op: BinOp, l: Ast, r: Ast, loc: Loc) -> Self {
        Self::new(
            AstKind::BinOp {
                op,
                l: Box::new(l),
                r: Box::new(r),
            },
            loc,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ParseError {
    /// 予期しないトークンがきた
    UnexpectedToken(Token),
    /// 式を期待していたのに式でないものがきた
    NotExpression(Token),
    /// 演算子を期待していたのに演算子でないものがきた
    NotOperator(Token),
    /// 括弧が閉じられていない
    UnclosedOpenParen(Token),
    /// 式の解析が終わったのにまだトークンが残っている
    RedundantExpression(Token),
    /// パース途中で入力が終わった
    Eof,
}

use std::iter::Peekable;
// use crate::AstKind::{BinOp, UniOp};

fn parse_left_binop<Tokens>(
    tokens: &mut Peekable<Tokens>,
    subexpr_parser: fn(&mut Peekable<Tokens>) -> Result<Ast, ParseError>,
    op_parser: fn(&mut Peekable<Tokens>) -> Result<BinOp, ParseError>,
) -> Result<Ast, ParseError>
where Tokens: Iterator<Item = Token>,
{
    let mut e= subexpr_parser(tokens)?;
    loop {
        match tokens.peek() {
            Some(_) => {
                let op = match op_parser(tokens) {
                    Ok(op) => op,
                    // ここでパースに失敗したのはこれ以上中置演算子がないという意味
                    Err(_) => break,
                };
                let r = subexpr_parser(tokens)?;
                let loc = e.loc.merge(&r.loc);
                e = Ast::binop(op, e, r, loc)
            }
            _ => break,
        }
    }
    Ok(e)
}

// primary = unumber
//       | "(" expr3 ")"
fn primary<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<Ast, ParseError>
where Tokens: Iterator<Item = Token>,
{
    tokens
        .next()
        .ok_or(ParseError::Eof)
        .and_then(|tok| match tok.value {
            // number
            TokenKind::Number(n) => Ok(Ast::new(AstKind::Num(n), tok.loc)),
            // | "(", EXPR3, ")" ;
            TokenKind::LParen => {
                let e = parse_expr(tokens)?;
                match tokens.next() {
                    Some(Token {
                             value: TokenKind::RParen,
                             ..
                         }) => Ok(e),
                    Some(t) => Err(ParseError::RedundantExpression(t)),
                    _ => Err(ParseError::UnclosedOpenParen(tok)),
                }
            }
            _ => Err(ParseError::NotExpression(tok)),
        })
}


// unary = ("+" | "-") primary
//        | primary
fn unary<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<Ast, ParseError>
where Tokens: Iterator<Item = Token>,
{
    match tokens.peek().map(|tok| tok.value) {
        Some(TokenKind::Plus) | Some(TokenKind::Minus) => {
            let op = match tokens.next() {
                Some(Token {
                    value: TokenKind::Plus,
                    loc,
                     }) => UniOp::plus(loc),
                Some(Token {
                    value: TokenKind::Minus,
                    loc,
                     }) => UniOp::minus(loc),
                _ => unreachable!(),
            };
            let e = primary(tokens)?;
            let loc = op.loc.merge(&e.loc);
            Ok(Ast::uniop(op, e, loc))
        }
        _ => primary(tokens),
    }
}

// mul = unary ("*" unary| "/" unary)*
fn mul<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<Ast, ParseError>
    where
        Tokens: Iterator<Item = Token>,
{
    // `parse_left_binop` に渡す関数を定義する
    fn parse_mul_op<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<BinOp, ParseError>
        where
            Tokens: Iterator<Item = Token>,
    {
        let op = tokens
            .peek()
            .ok_or(ParseError::Eof)
            .and_then(|tok| match tok.value {
                TokenKind::Asterisk => Ok(BinOp::mult(tok.loc.clone())),
                TokenKind::Slash => Ok(BinOp::div(tok.loc.clone())),
                _ => Err(ParseError::NotOperator(tok.clone())),
            })?;
        tokens.next();
        Ok(op)
    }

    parse_left_binop(tokens, unary, parse_mul_op)
}

// add = mul ("+" mul | "-" mul)*
fn add<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<Ast, ParseError>
where Tokens: Iterator<Item = Token>,
{
    fn parse_add_op<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<BinOp, ParseError>
    where Tokens: Iterator<Item = Token>,
    {
        let op = tokens
            .peek()
            .ok_or(ParseError::Eof)
            .and_then(|tok| match tok.value {
                TokenKind::Plus => Ok(BinOp::add(tok.loc.clone())),
                TokenKind::Minus => Ok(BinOp::sub(tok.loc.clone())),
                _ => Err(ParseError::NotOperator(tok.clone())),
            })?;
        tokens.next();
        Ok(op)
    }

    parse_left_binop(tokens, mul, parse_add_op)
}

fn relational<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<Ast, ParseError>
    where Tokens: Iterator<Item = Token>,
{
    fn parse_relational_op<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<BinOp, ParseError>
        where Tokens: Iterator<Item = Token>,
    {
        let op = tokens
            .peek()
            .ok_or(ParseError::Eof)
            .and_then(|tok| match tok.value {
                TokenKind::Lt => Ok(BinOp::lt(tok.loc.clone())),
                TokenKind::Le => Ok(BinOp::le(tok.loc.clone())),
                _ => Err(ParseError::NotOperator(tok.clone())),
            })?;
        tokens.next();
        Ok(op)
    }

    parse_left_binop(tokens, add, parse_relational_op)
}

// equality   = relational ("==" relational | "!=" relational)*
fn parse_equality<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<Ast, ParseError>
where Tokens: Iterator<Item = Token>,
{
    fn parse_eq_op<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<BinOp, ParseError>
        where Tokens: Iterator<Item = Token>,
    {
        let op = tokens
            .peek()
            .ok_or(ParseError::Eof)
            .and_then(|tok| match tok.value {
                TokenKind::Eq => Ok(BinOp::eq(tok.loc.clone())),
                TokenKind::Neq => Ok(BinOp::neq(tok.loc.clone())),
                _ => Err(ParseError::NotOperator(tok.clone())),
            })?;
        tokens.next();
        Ok(op)
    }

    parse_left_binop(tokens, relational, parse_eq_op)
}

fn parse_expr<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<Ast, ParseError>
    where
        Tokens: Iterator<Item = Token>,
{
    parse_equality(tokens)
}

fn parse(tokens: Vec<Token>) -> Result<Ast, ParseError> {
    let mut tokens = tokens.into_iter().peekable();
    let ret = parse_expr(&mut tokens)?;
    match tokens.next() {
        Some(tok) => Err(ParseError::RedundantExpression(tok)),
        None => Ok(ret),
    }
}

#[test]
fn test_parser() {
    // 1 + 2 * 3 - -10
    let ast = parse(vec![
        Token::number(1, Loc(0, 1)),
        Token::plus(Loc(2, 3)),
        Token::number(2, Loc(4, 5)),
        Token::asterisk(Loc(6, 7)),
        Token::number(3, Loc(8, 9)),
        Token::minus(Loc(10, 11)),
        Token::minus(Loc(12, 13)),
        Token::number(10, Loc(13, 15)),
    ]);
    assert_eq!(
        ast,
        Ok(Ast::binop(
            BinOp::sub(Loc(10, 11)),
            Ast::binop(
                BinOp::add(Loc(2, 3)),
                Ast::num(1, Loc(0, 1)),
                Ast::binop(
                    BinOp::new(BinOpKind::Mult, Loc(6, 7)),
                    Ast::num(2, Loc(4, 5)),
                    Ast::num(3, Loc(8, 9)),
                    Loc(4, 9)
                ),
                Loc(0, 9),
            ),
            Ast::uniop(
                UniOp::minus(Loc(12, 13)),
                Ast::num(10, Loc(13, 15)),
                Loc(12, 15)
            ),
            Loc(0, 15)
        ))
    )
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

/// 字句解析エラーと構文解析エラーを統合するエラー型
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Error {
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
            Eq => write!(f, "=="),
            Neq => write!(f, "!="),
            Le => write!(f, "<="),
        }
    }
}

impl fmt::Display for Loc {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}-{}", self.0, self.1)
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
    fn show_diagnostic(&self, input: &str) {
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

fn show_trace<E: StdError>(e: E) {
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

struct RpnCompiler;

impl RpnCompiler {
    pub fn new() -> Self {
        RpnCompiler
    }

    pub fn compile(&mut self, expr: &Ast) -> String {
        let mut buf = String::new();
        buf.push_str(".intel_syntax noprefix\n");
        buf.push_str(".globl main\n");
        buf.push_str("main:\n");

        self.compile_inner(expr, &mut buf);

        buf.push_str("  pop rax\n");
        buf.push_str("  ret\n");
        buf
    }

    pub fn compile_inner(&mut self, expr: &Ast, buf: &mut String) {
        use self::AstKind::*;

        match expr.value {
            Num(n) => {
                buf.push_str(&format!("  push {}\n", n));
            },
            UniOp { ref op, ref e} => {
                // self.compile_uniop(op, buf);
                // self.compile_inner(e, buf);
                match op.value {
                    UniOpKind::Plus => {
                        self.compile_inner(e, buf);
                    }
                    UniOpKind::Minus => {
                        self.compile_inner(e, buf);
                        buf.push_str("  pop rax\n");
                        buf.push_str("  neg rax\n");
                        buf.push_str("  push rax\n");
                    }
                }
            }
            BinOp {
                ref op,
                ref l,
                ref r,
            } => {
                self.compile_inner(l, buf);
                self.compile_inner(r, buf);
                buf.push_str("  pop rdi\n");
                buf.push_str("  pop rax\n");

                match op.value {
                    BinOpKind::Add => {
                        buf.push_str("  add rax, rdi\n");
                    }
                    BinOpKind::Sub => {
                        buf.push_str("  sub rax, rdi\n");
                    }
                    BinOpKind::Mult => {
                        buf.push_str("  imul rax, rdi\n");
                    }
                    BinOpKind::Div => {
                        buf.push_str("  cqo\n");
                        buf.push_str("  idiv rdi\n");
                    }
                    BinOpKind::Eq => {
                        buf.push_str("  cmp rax, rdi\n");
                        buf.push_str("  sete al\n");
                        buf.push_str("  movzb rax, al\n");
                    }
                    BinOpKind::Neq => {
                        buf.push_str("  cmp rax, rdi\n");
                        buf.push_str("  setne al\n");
                        buf.push_str("  movzb rax, al\n");
                    }
                    BinOpKind::Lt => {
                        buf.push_str("  cmp rax, rdi\n");
                        buf.push_str("  setl al\n");
                        buf.push_str("  movzb rax, al\n");
                    }
                    BinOpKind::Le => {
                        buf.push_str("  cmp rax, rdi\n");
                        buf.push_str("  setle al\n");
                        buf.push_str("  movzb rax, al\n");
                    }
                }

                buf.push_str("  push rax\n");
            }
        }
    }
}

use std::io;
use std::io::Write;

fn prompt(s: &str) -> io::Result<()> {
    use std::io::{stdout, Write};
    let stdout = stdout();
    let mut stdout = stdout.lock();
    stdout.write(s.as_bytes())?;
    stdout.flush()
}

fn main() {
    use std::io::{stdin, BufRead, BufReader};
    let mut compiler = RpnCompiler::new();

    let stdin = stdin();
    let stdin = stdin.lock();
    let stdin = BufReader::new(stdin);
    let mut lines = stdin.lines();

    loop {
        // prompt("> ").unwrap();
        if let Some(Ok(line)) = lines.next() {
            let ast = match line.parse::<Ast>() {
                Ok(ast) => ast,
                Err(e) => {
                    e.show_diagnostic(&line);
                    show_trace(e);
                    continue;
                }
            };
            // println!("{:?}", ast);
            let rpn = compiler.compile(&ast);
            println!("{}", rpn);
        } else {
            break;
        }
    }
}
