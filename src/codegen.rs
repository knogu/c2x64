use crate::parse::*;

pub struct AsmGenerator;

impl AsmGenerator {
    pub fn new() -> Self {
        AsmGenerator
    }

    pub fn codegen(&mut self, stmt_asts: Vec<Ast>) -> String {
        let mut buf = String::new();
        buf.push_str(".intel_syntax noprefix\n");
        buf.push_str(".globl main\n");
        buf.push_str("main:\n");

        for stmt_ast in &stmt_asts {
            self.codegen_inner(stmt_ast, &mut buf);
        }

        buf.push_str("  ret\n");
        buf
    }

    pub fn codegen_inner(&mut self, expr: &Ast, buf: &mut String) {
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
                        self.codegen_inner(e, buf);
                    }
                    UniOpKind::Minus => {
                        self.codegen_inner(e, buf);
                        buf.push_str("  pop rax\n");
                        buf.push_str("  neg rax\n");
                        buf.push_str("  push rax\n");
                    }
                }
            },
            Return {ref e} => {
                self.codegen_inner(e, buf);
                buf.push_str("  pop rax\n");
                buf.push_str("  ret\n");
            }
            ExpressionStmt {ref e} => {
                self.codegen_inner(e, buf);
                buf.push_str("  add rsp, 8\n");
            }
            BinOp {
                ref op,
                ref l,
                ref r,
            } => {
                self.codegen_inner(l, buf);
                self.codegen_inner(r, buf);
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
                    BinOpKind::Gt => {
                        buf.push_str("  cmp rax, rdi\n");
                        buf.push_str("  setg al\n");
                        buf.push_str("  movzb rax, al\n");
                    }
                    BinOpKind::Ge => {
                        buf.push_str("  cmp rax, rdi\n");
                        buf.push_str("  setge al\n");
                        buf.push_str("  movzb rax, al\n");
                    }
                }

                buf.push_str("  push rax\n");
            }
        }
    }
}
