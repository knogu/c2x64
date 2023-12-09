.intel_syntax noprefix
.globl main
main:
  push 1
  add rsp, 8
  push 2
  add rsp, 8
  push 3
  pop rax
  ret
  ret

