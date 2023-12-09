mod tokenize;
mod parse;
mod codegen;
mod error;

use crate::parse::*;
use crate::codegen::*;
use crate::error::*;

use std::error::Error as StdError;
use std::str::FromStr;
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
    let mut asm_generator = AsmGenerator::new();

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
            let asm = asm_generator.codegen(&ast);
            println!("{}", asm);
        } else {
            break;
        }
    }
}
