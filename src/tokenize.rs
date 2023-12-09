#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Loc(pub(crate) usize, pub(crate) usize);

impl Loc {
    pub(crate) fn merge(&self, other: &Loc) -> Loc {
        use std::cmp::{max, min};
        Loc(min(self.0, other.0), max(self.1, other.1))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Annot<T> {
    pub(crate) value: T,
    pub(crate) loc: Loc,
}

impl<T> Annot<T> {
    pub(crate) fn new(value: T, loc: Loc) -> Self {
        Self {value, loc}
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenKind {
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
    Gt, // Greater than
    Ge, // Greater or Equal
    Semicolon,
    Return,
}

pub type Token = Annot<TokenKind>;

impl Token {
    pub(crate) fn number(n: u64, loc: Loc) -> Self {
        Self::new(TokenKind::Number(n), loc)
    }

    pub(crate) fn plus(loc: Loc) -> Self {
        Self::new(TokenKind::Plus, loc)
    }

    pub(crate) fn minus(loc: Loc) -> Self {
        Self::new(TokenKind::Minus, loc)
    }

    pub(crate) fn asterisk(loc: Loc) -> Self {
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

    fn ge(loc: Loc) -> Self {
        Self::new(TokenKind::Ge, loc)
    }

    fn gt(loc: Loc) -> Self {
        Self::new(TokenKind::Gt, loc)
    }

    fn semicolon(loc: Loc) -> Self {
        Self::new(TokenKind::Semicolon, loc)
    }

    fn return_(loc: Loc) -> Self {
        Self::new(TokenKind::Return, loc)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LexErrorKind {
    InvalidChar(char),
    Eof,
}

pub type LexError = Annot<LexErrorKind>;

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

fn lex_gt(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    consume_byte(input, start, b">").map(|(_, end)| (Token::gt(Loc(start, end)), end))
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

fn lex_ge(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    consume_byte(input, start, b">=").map(|(_, end)| (Token::ge(Loc(start, end)), end))
}

fn lex_semicolon(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    consume_byte(input, start, b";").map(|(_, end)| (Token::semicolon(Loc(start, end)), end))
}

fn lex_return(input: &[u8], start: usize) -> Result<(Token, usize), LexError> {
    consume_byte(input, start, b"return").map(|(_, end)| (Token::return_(Loc(start, end)), end))
}

fn skip_spaces(input: &[u8], pos: usize) -> Result<((), usize), LexError> {
    let pos = recognize_many(input, pos, |b| b" \n\t".contains(&b));
    Ok(((), pos))
}

fn starts_with(input: &[u8], with: &[u8], pos: usize) -> bool {
    let size = with.len();
    if pos + size - 1 >= input.len() {
        return false
    }
    return &input[pos..pos + size] == with
}

pub fn lex(input: &str) -> Result<Vec<Token>, LexError> {
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
        if starts_with(input, b"return", pos) {
            lex_a_token!(lex_return(input, pos));
            continue
        }
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
            if &input[pos..pos+2] == b">=" {
                lex_a_token!(lex_ge(input, pos));
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
            b'>' => lex_a_token!(lex_gt(input, pos)),
            b';' => lex_a_token!(lex_semicolon(input, pos)),
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

