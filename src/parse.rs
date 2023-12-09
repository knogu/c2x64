use crate::tokenize::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum UniOpKind {
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
pub enum BinOpKind {
    Add,
    Sub,
    Mult,
    Div,
    Eq,
    Neq,
    Lt,
    Le,
    Gt,
    Ge,
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

    fn gt(loc: Loc) -> Self {
        Self::new(BinOpKind::Gt, loc)
    }

    fn ge(loc: Loc) -> Self {
        Self::new(BinOpKind::Ge, loc)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AstKind {
    Num(u64),
    UniOp {op: UniOp, e: Box<Ast>},
    BinOp {op: BinOp, l: Box<Ast>, r: Box<Ast>},
}

pub type Ast = Annot<AstKind>;

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
pub enum ParseError {
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
    SemicolonNotFound,
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

// relational = add ("<" add | "<=" add | ">" add | ">=" add)*
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
                TokenKind::Gt => Ok(BinOp::gt(tok.loc.clone())),
                TokenKind::Ge => Ok(BinOp::ge(tok.loc.clone())),
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

fn parse_stmt<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<Ast, ParseError>
    where
        Tokens: Iterator<Item = Token>,
{
    let res = parse_expr(tokens)?;
    match tokens.next() {
        Some(Token {
                 value: TokenKind::Semicolon,
                 ..
             }) => { return Ok(res) },
        _ => Err(ParseError::SemicolonNotFound),
   }
}

fn program<Tokens>(tokens: &mut Peekable<Tokens>) -> Result<Vec<Ast>, ParseError>
where
Tokens: Iterator<Item = Token>,
{
    let mut stmts = vec![];
    loop {
        match tokens.peek() {
            Some(_) => {
                stmts.push(parse_stmt(tokens)?);
            }
            _ => break,
        }
    }
    return Ok(stmts);
}

pub fn parse(tokens: Vec<Token>) -> Result<Vec<Ast>, ParseError> {
    let mut tokens = tokens.into_iter().peekable();
    let ret = program(&mut tokens)?;
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
